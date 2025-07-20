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

genai.configure(api_key=st.secrets["api"]["gemini_key"])
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

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
                
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"### {result['title']}\n\n"
                    f"{result['body']}\n\n"
                    f"[Read more]({result['href']})"
                )
            
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
    if "query_cache" not in st.session_state:
        st.session_state["query_cache"] = {}

    cache_key = question.strip().lower()
    if cache_key in st.session_state["query_cache"]:
        cached_answer = st.session_state["query_cache"][cache_key]
        if not cached_answer.startswith("[cached]"):
            return f"[cached] {cached_answer}"
        return cached_answer
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=10)
    docs = [doc for doc, _ in docs_and_scores]
    
    chain = conversational_chain()
    response = chain({
        "input_documents": docs,
        "question": question,
        "chat_history": chat_history
    }, return_only_outputs=True)

    if "WEB_SEARCH_FALLBACK" in response["output_text"]:
        web_response = get_web_response(question, chat_history)
        st.session_state["query_cache"][cache_key] = web_response
        return web_response if web_response else response["output_text"]
    
    st.session_state["query_cache"][cache_key] = response["output_text"]
    return response["output_text"]


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
</style>
""",
        unsafe_allow_html=True,
    )

def main():
    set_custom_style()
    st.set_page_config(page_title="Dbaas - ChatPDF", page_icon="📄")

    st.markdown('<div class="title-bar">💬 Dbaas - ChatPDF</div>', unsafe_allow_html=True)

    if "pdfs" not in st.session_state or not st.session_state["pdfs"]:
        st.markdown("📂 Upload docs using the button in sidebar. Works on both Text PDF and Scanned PDF (OCR Included)")

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

        st.markdown("---")

    if "pdfs" in st.session_state and st.session_state["pdfs"] and process:
        with st.spinner("📝 Processing PDFs and building index..."):
            raw_text = extract_texts_from_pdfs(st.session_state["pdfs"])
            chunks = make_chunks(raw_text)
            if not chunks:
                st.error("No text chunks generated from the uploaded PDFs. PDF may consist disorted Photos, which can't be extracted by OCR.")
                return
            vector = vectorization(chunks)
            if vector is None:
                return
        st.session_state.processed = True

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
