import io
from ddgs import DDGS
import streamlit as st
from PyPDF2 import PdfReader
import pytesseract
import fitz
from PIL import Image
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# import os
# from dotenv import load_dotenv
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# genai.configure(api_key=st.secrets["api"]["gemini_key"])

# Google API Initialization requires API to be set up securely using os.

import os
os.environ["GOOGLE_API_KEY"] = st.secrets["api"]["gemini_key"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

class ConversationMemory:
    def __init__(self):
        self.conversation_chunks = []
        self.conversation_vector = None
        self.embeddings_model = None
    
    def initialize_embeddings(self):
        """Initialize embeddings model if not already done"""
        if self.embeddings_model is None:
            self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def add_exchange(self, question, answer):
        """Add Q&A pair to conversation memory"""
        # Clean answer (remove cache markers and truncate if too long)
        clean_answer = answer.replace("[cached]", "").strip()
        if len(clean_answer) > 800:
            clean_answer = clean_answer[:800] + "..."
        
        # Create exchange chunk
        exchange = f"Previous Question: {question}\nPrevious Answer: {clean_answer}"
        self.conversation_chunks.append(exchange)
        
        # Keep only last 15 exchanges to prevent memory bloat
        if len(self.conversation_chunks) > 15:
            self.conversation_chunks = self.conversation_chunks[-15:]
        
        # Rebuild conversation vector store
        self._rebuild_conversation_vector()
    
    def _rebuild_conversation_vector(self):
        """Rebuild the conversation vector store with current chunks"""
        if not self.conversation_chunks:
            self.conversation_vector = None
            return
        
        try:
            self.initialize_embeddings()
            self.conversation_vector = FAISS.from_texts(
                self.conversation_chunks, 
                embedding=self.embeddings_model
            )
        except Exception as e:
            st.error(f"Error building conversation vector: {str(e)}")
            self.conversation_vector = None
    
    def get_relevant_context(self, current_question, top_k=3):
        """Retrieve relevant past conversations"""
        if not self.conversation_vector or not self.conversation_chunks:
            return ""
        
        try:
            relevant_convs = self.conversation_vector.similarity_search(current_question, k=top_k)
            context_parts = []
            
            for i, doc in enumerate(relevant_convs, 1):
                context_parts.append(f"Relevant Past Exchange {i}:\n{doc.page_content}")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            st.error(f"Error retrieving conversation context: {str(e)}")
            return ""

def extract_texts_from_pdfs(pdfs):
    texts = ""

    for pdf in pdfs:
        if hasattr(pdf, "seek"):
            pdf.seek(0)
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                texts += text + "\n"

    if not texts.strip():
        for pdf in pdfs:
            if hasattr(pdf, "seek"):
                pdf.seek(0)
            pdf_bytes = pdf.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img)
                texts += ocr_text + "\n"

    return texts

def make_chunks(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(texts)

def vectorization(chunks):
    if not chunks:
        st.error("No text chunks to embed. Please upload PDFs with extractable text. (Try uploading Text PDF or Properly Scanned PDF)")
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector = FAISS.from_texts(chunks, embedding=embeddings)
    except IndexError:
        st.error("Vectorization failed: received empty embeddings.")
        return None
    vector.save_local("faiss_index")
    return vector

def get_contextualized_query(question, chat_history, max_history=4):
    """Expand current query with relevant context from chat history"""
    if not chat_history:
        return question
    
    # Get last few exchanges for context
    recent_context = []
    for msg in chat_history[-max_history*2:]:  # *2 because user+assistant pairs
        if msg['role'] == 'user':
            recent_context.append(f"Recent question: {msg['content']}")
        elif msg['role'] == 'assistant' and not msg['content'].startswith('[cached]'):
            # Extract key points from previous answers (first 150 chars)
            clean_answer = msg['content'].replace("[cached]", "").strip()
            recent_context.append(f"Recent context: {clean_answer[:150]}...")
    
    if recent_context:
        context_str = "\n".join(recent_context[-4:])  # Last 4 context items
        expanded_query = f"{question}\n\nConversation context:\n{context_str}"
        return expanded_query
    
    return question

def conversational_chain_enhanced():
    """Enhanced prompt template with conversation memory"""
    prompt_template = """
    You are a helpful assistant answering questions about uploaded documents. Use the provided context to give accurate, detailed responses.

    DOCUMENT CONTEXT:
    {context}

    RELEVANT PAST CONVERSATIONS:
    {conversation_context}

    RECENT CHAT HISTORY:
    {chat_history}

    CURRENT QUESTION: {question}

    Instructions:
    1. First check if the question can be answered using the DOCUMENT CONTEXT
    2. Use RELEVANT PAST CONVERSATIONS to understand context and avoid repeating information
    3. Reference previous discussions when relevant (e.g., "As we discussed earlier..." or "Building on our previous conversation...")
    4. If you cannot find the answer in the provided documents, respond with "WEB_SEARCH_FALLBACK"
    5. Be conversational and acknowledge the ongoing discussion flow
    6. Provide detailed, comprehensive answers when possible

    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question", "chat_history", "conversation_context"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def format_recent_history(chat_history, max_exchanges=2):
    """Format recent chat history for prompt"""
    if not chat_history:
        return "No previous conversation."
    
    formatted = []
    recent_messages = chat_history[-(max_exchanges*2):]  # Get last N exchanges
    
    for msg in recent_messages:
        role = "Human" if msg['role'] == 'user' else "Assistant"
        content = msg['content'].replace("[cached]", "").strip()
        if len(content) > 200:
            content = content[:200] + "..."
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)

def get_web_response(question, chat_history):
    try:
        with DDGS() as ddgs:
            last_3_messages = "\n".join(
                [f"{msg['role']}: {msg['content']}" 
                 for msg in chat_history[-3:] if msg['role'] == "user"]
            )
            enhanced_query = f"{question}\n\nRelevant conversation history:\n{last_3_messages}"
            
            results = [r for r in ddgs.text(enhanced_query, max_results=1)]
            
            if not results:
                return None
                
            formatted_response = (
                "⚠️ Could not find in PDFs. Here are relevant web results:\n\n"
                f"*{results[0]['body']}*\n\n"
                f"Source: [{results[0]['title']}]({results[0]['href']})"
            )
            
            return formatted_response
            
    except Exception as e:
        print(f"Web search failed: {str(e)}")
        return None

def get_bot_response(question, chat_history):
    """Enhanced context-aware response generation with conversation memory"""
    
    # Initialize session state components
    if "query_cache" not in st.session_state:
        st.session_state["query_cache"] = {}
    if "conversation_memory" not in st.session_state:
        st.session_state["conversation_memory"] = ConversationMemory()

    # Check cache first (with contextualized key)
    cache_key = f"{question.strip().lower()}_{len(chat_history)}"
    if cache_key in st.session_state["query_cache"]:
        cached_answer = st.session_state["query_cache"][cache_key]
        if not cached_answer.startswith("[cached]"):
            return f"[cached] {cached_answer}"
        return cached_answer
    
    # 1. Create contextualized query for better retrieval
    contextualized_query = get_contextualized_query(question, chat_history)
    
    # 2. Search document vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Search documents with contextualized query
    docs_and_scores = vectorstore.similarity_search_with_score(contextualized_query, k=8)
    docs = [doc for doc, _ in docs_and_scores]
    
    # 3. Get relevant conversation context
    conv_memory = st.session_state["conversation_memory"]
    relevant_conv_context = conv_memory.get_relevant_context(question, top_k=3)
    
    # 4. Generate response with enhanced context
    chain = conversational_chain_enhanced()
    response = chain({
        "input_documents": docs,
        "question": question,
        "chat_history": format_recent_history(chat_history, max_exchanges=2),
        "conversation_context": relevant_conv_context if relevant_conv_context else "No relevant past conversations."
    }, return_only_outputs=True)

    # 5. Handle fallback to web search
    if "WEB_SEARCH_FALLBACK" in response["output_text"]:
        web_response = get_web_response(question, chat_history)
        final_response = web_response if web_response else response["output_text"]
    else:
        final_response = response["output_text"]
    
    # 6. Update conversation memory and cache
    conv_memory.add_exchange(question, final_response)
    st.session_state["query_cache"][cache_key] = final_response
    
    return final_response

def set_custom_style():
    st.markdown(
        """
<style>
    .title-bar {
        background-color: #00755E;
        color: white;
        font-size: 28px;
        font-weight: 700;
        padding: 14px 24px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem;
        user-select: none;
        border-radius: 6px;
    }
    .stApp {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00755E !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] {
        background-color: #000000;
        color: #00755E;
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] ::-webkit-scrollbar {
        width: 8px;
    }
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
        background-color: #004c40;
        border-radius: 4px;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, strong {
        color: #00755E !important;
        font-weight: 700;
    }
    .stTextInput>div>div>input, .stFileUploader>div>div {
        background-color: #111111;
        color: #ffffff;
        border: 1.5px solid #00755E;
        border-radius: 6px;
    }
    .stButton>button {
        background-color: #00755E;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 18px;
        font-size: 16px;
        font-weight: 600;
        transition: background-color 0.2s ease;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #005a43;
        color: white;
    }
    .stSpinner>div>div {
        border-color: #00755E transparent transparent transparent !important;
    }
    .stChatInput>div>div>textarea {
        background-color: #111111;
        color: white;
        border: 1.5px solid #00755E;
        border-radius: 8px;
        font-size: 16px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 12px;
        resize: vertical;
        min-height: 50px;
    }
    .context-info {
        background-color: #1a1a1a;
        border-left: 3px solid #00755E;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
        font-size: 12px;
        color: #cccccc;
    }
</style>
""",
        unsafe_allow_html=True,
    )

def main():
    set_custom_style()
    st.set_page_config(page_title="Bala - Context-Aware ChatPDFs", page_icon="📄")

    st.markdown('<div class="title-bar">💬 Bala - Context-Aware ChatPDFs</div>', unsafe_allow_html=True)

    if "pdfs" not in st.session_state or not st.session_state["pdfs"]:
        st.markdown("""
                    📂 Add docs via **sidebar**\n
                    🔍 Handles **Text** & **Scanned** PDFs (OCR inside)\n
                    🧠 **Context aware** responses with conversation memory\n
                    🔄 **Remembers** previous questions and answers\n
                    🚫 **No SignUp** required\n
                    🛡️ Your docs are **never saved** anywhere online\n
                    """)
        
    with st.sidebar:
        st.header("📁 Upload PDFs")
        uploaded_pdfs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_pdfs:
            existing_pdfs = st.session_state.get("pdfs", [])            
            existing_filenames = {pdf.name for pdf in existing_pdfs}
            new_files = [pdf for pdf in uploaded_pdfs if pdf.name not in existing_filenames]
            combined = existing_pdfs + new_files
            st.session_state["pdfs"] = combined
            st.markdown(f"**{len(combined)} file(s) selected**")
        else:
            if "pdfs" in st.session_state:
                del st.session_state["pdfs"]

        process = st.button("Process Documents", key="process_button")

        if "processed" in st.session_state and st.session_state.processed:
            st.success("✅ Documents processed and indexed!")
            
            # Show conversation memory stats
            if "conversation_memory" in st.session_state:
                conv_memory = st.session_state["conversation_memory"]
                num_exchanges = len(conv_memory.conversation_chunks)
                if num_exchanges > 0:
                    st.info(f"💭 Conversation memory: {num_exchanges} exchanges stored")

        st.markdown("---")
        
        # Add conversation memory controls
        if "conversation_memory" in st.session_state and st.session_state["conversation_memory"].conversation_chunks:
            st.subheader("🧠 Memory Controls")
            if st.button("Clear Conversation Memory", key="clear_memory"):
                st.session_state["conversation_memory"] = ConversationMemory()
                st.session_state["query_cache"] = {}
                st.success("Memory cleared!")
                st.rerun()

    if "pdfs" in st.session_state and st.session_state["pdfs"] and process:
        with st.spinner("📝 Processing PDFs and building index..."):
            raw_text = extract_texts_from_pdfs(st.session_state["pdfs"])
            chunks = make_chunks(raw_text)
            if not chunks:
                st.error("No text chunks generated from the uploaded PDFs. PDF may consist distorted Photos, which can't be extracted by OCR.")
                return
            vector = vectorization(chunks)
            if vector is None:
                return
        st.session_state.processed = True

        # Reset conversation memory when new documents are processed
        st.session_state["conversation_memory"] = ConversationMemory()
        st.session_state["query_cache"] = {}

        if "summary_request_count" not in st.session_state:
            st.session_state.summary_request_count = 1
        else:
            st.session_state.summary_request_count += 1

        summary_question = f"summarize the document (request {st.session_state.summary_request_count})"

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "content": summary_question})
        with st.spinner("🧠 Generating summary..."):
            answer = get_bot_response(summary_question, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        role = chat["role"]
        content = chat["content"]
        prefix = "👤" if role == "user" else "🤖"
        with st.chat_message(role):
            st.markdown(f"{prefix}  {content}")

    user_input = st.chat_input("Ask a question about the PDF(s):")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(f"👤  {user_input}")
        with st.spinner("🧠 Thinking..."):
            answer = get_bot_response(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(f"🤖  {answer}")

if __name__ == "__main__":
    main()
