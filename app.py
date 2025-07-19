import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def extract_texts_from_pdfs(pdfs):
    texts = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts += text + "\n"
    return texts


def make_chunks(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(texts)


def vectorization(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector = FAISS.from_texts(chunks, embedding=embeddings)
    vector.save_local("faiss_index")


def conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible using the provided context and previous conversation history.
    If the answer is not in the provided context, just say "WEB_SEARCH_FALLBACK".
    Previous Conversation:{chat_history}
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def get_web_response(question, chat_history):
    """Get web results with chat history context"""
    try:
        with DDGS() as ddgs:
            last_3_messages = "\n".join(
                [f"{msg['role']}: {msg['content']}" 
                 for msg in chat_history[-3:] if msg['role'] == "user"]
            )
            enhanced_query = f"{question}\n\nRelevant conversation history:\n{last_3_messages}"
            
            results = [r for r in ddgs.text(enhanced_query, max_results=3)]
            
            if not results:
                return None
                
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"### {result['title']}\n\n"
                    f"{result['body']}\n\n"
                    f"[Read more]({result['href']})"
                )
            
            return (
                "⚠️ Could not find in PDFs. Here are relevant web results:\n\n"
                + "\n\n---\n\n".join(formatted_results)
            )
            
    except Exception as e:
        print(f"Web search failed: {str(e)}")
        return None

def should_use_web_fallback(docs_and_scores):
    if not docs_and_scores:
        return True
        
    avg_score = sum(score for _, score in docs_and_scores) / len(docs_and_scores)
    return avg_score < 0.5

def get_bot_response(question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=4)
    docs = [doc for doc, _ in docs_and_scores]

    if should_use_web_fallback(docs_and_scores):
        web_response = get_web_response(question, chat_history)
        return web_response if web_response else "I couldn't find relevant information."

    chain = conversational_chain()
    response = chain({
        "input_documents": docs,
        "question": question,
        "chat_history": chat_history
    }, return_only_outputs=True)

    if "WEB_SEARCH_FALLBACK" in response["output_text"]:
        web_response = get_web_response(question, chat_history)
        return web_response if web_response else response["output_text"]
    
    return response["output_text"]


def set_custom_style():
    st.markdown(
        """
<style>
    /* Title bar styling */
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

    /* Overall background and text */
    .stApp {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Headers color */
    h1, h2, h3, h4, h5, h6 {
        color: #00755E !important;
        font-weight: 700;
    }

    /* Sidebar background and text */
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

    /* Sidebar headers */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, strong {
        color: #00755E !important;
        font-weight: 700;
    }

    /* Input and uploader styling */
    .stTextInput>div>div>input, .stFileUploader>div>div {
        background-color: #111111;
        color: #ffffff;
        border: 1.5px solid #00755E;
        border-radius: 6px;
    }

    /* Button styling */
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

    /* Spinner color */
    .stSpinner>div>div {
        border-color: #00755E transparent transparent transparent !important;
    }

    /* Chat input styling */
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
</style>
""",
        unsafe_allow_html=True,
    )


def main():
    set_custom_style()
    st.set_page_config(page_title="Dbaas - ChatPDF", page_icon="📄")

    st.markdown('<div class="title-bar">💬 Dbaas - ChatPDF</div>', unsafe_allow_html=True)

    # Message before upload
    if "pdfs" not in st.session_state or not st.session_state["pdfs"]:
        st.markdown("📂 Upload docs using the button in sidebar")

    with st.sidebar:
        st.header("📁 Upload PDFs")
        uploaded_pdfs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_pdfs:
            st.session_state["pdfs"] = uploaded_pdfs
            st.markdown(f"**{len(uploaded_pdfs)} file(s) selected**")
        else:
            if "pdfs" in st.session_state:
                del st.session_state["pdfs"]

        process = st.button("Process Documents", key="process_button")

        if "processed" in st.session_state and st.session_state.processed:
            st.success("✅ Documents processed and indexed!")

        st.markdown("---")

    # Process PDFs after pressing button
    if "pdfs" in st.session_state and st.session_state["pdfs"] and process:
        with st.spinner("📝 Processing PDFs and building index..."):
            raw_text = extract_texts_from_pdfs(st.session_state["pdfs"])
            chunks = make_chunks(raw_text)
            vectorization(chunks)
        st.session_state.processed = True

        summary_question = "summarize the pdf"
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "content": summary_question})
        with st.spinner("🧠 Generating summary..."):
            answer = get_bot_response(summary_question, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages
    for chat in st.session_state.chat_history:
        role = chat["role"]
        content = chat["content"]
        prefix = "👤" if role == "user" else "🤖"
        with st.chat_message(role):
            st.markdown(f"{prefix}  {content}")

    # User input
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