import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Ask From URL", layout="wide")

# ---------------------------
# API KEY (Use secrets in production)
# ---------------------------
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "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")

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ---------------------------
# UI TITLE
# ---------------------------
st.title("🌐 Ask Questions From Website")

# ---------------------------
# SIDEBAR INPUT
# ---------------------------
st.sidebar.header("🔗 Enter URLs")

urls = st.sidebar.text_area("Enter URLs (one per line)")
process_btn = st.sidebar.button("Process URLs")

# ---------------------------
# CREATE VECTOR STORE
# ---------------------------
@st.cache_resource
def create_vector_store(url_list):
    loader = UnstructuredURLLoader(urls=url_list)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(data)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_index = FAISS.from_documents(docs, embedding)
    return vector_index


# ---------------------------
# PROCESS URLS
# ---------------------------
if process_btn and urls:
    url_list = [u.strip() for u in urls.split("\n") if u.strip()]

    with st.spinner("🔄 Processing URLs..."):
        st.session_state.vectorstore = create_vector_store(url_list)

    st.sidebar.success("✅ URLs processed successfully!")

# ---------------------------
# MAIN QA SECTION
# ---------------------------
if "vectorstore" in st.session_state:
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    st.subheader("💬 Ask your question")

    # ---------------------------
    # FORM (Enter + Button)
    # ---------------------------
    with st.form("qa_form"):
        question = st.text_input("Enter your question")
        submit = st.form_submit_button("Ask Me")

    if submit and question:
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided context.
            If the context is insufficient, say "I don't know".

            Context:
            {context}

            Question: {question}
            """,
            input_variables=['context', 'question']
        )

        with st.spinner("🔍 Retrieving answer..."):
            retrieved_docs = retriever.invoke(question)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            final_prompt = prompt.invoke({
                "context": context_text,
                "question": question
            })

            answer = llm.invoke(final_prompt)

        # ---------------------------
        # OUTPUT
        # ---------------------------
        st.subheader("📌 Answer")
        st.write(answer.content)

        # ---------------------------
        # SOURCE CONTEXT
        # ---------------------------
        with st.expander("📄 Source Context"):
            st.write(context_text)

else:
    st.info("👈 Enter URLs in the sidebar and click 'Process URLs' to begin.")
