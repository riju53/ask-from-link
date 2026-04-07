import os
import streamlit as st

# LangChain + Groq
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ----------------------
# 🔐 API KEY
# ----------------------
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"

# ----------------------
# UI
# ----------------------
st.set_page_config(page_title="Stock RAG App", layout="wide")
st.title("📈 Stock Research Assistant")

# Sidebar URLs
st.sidebar.subheader("🌐 Enter Stock URLs")

with st.sidebar.form("url_form"):
    url1 = st.text_input("URL 1")
    url2 = st.text_input("URL 2")
    url3 = st.text_input("URL 3")
    submit_urls = st.form_submit_button("Submit URLs")

# Save URLs
if submit_urls:
    st.session_state.urls = [u for u in [url1, url2, url3] if u]

# ----------------------
# LLM
# ----------------------
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# ----------------------
# Build Vector DB
# ----------------------
@st.cache_resource
def build_vector_db(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(data)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embedding)

    return vector_db

# Build retriever once
if "urls" in st.session_state and "retriever" not in st.session_state:
    with st.spinner("Processing URLs..."):
        db = build_vector_db(st.session_state.urls)
        st.session_state.retriever = db.as_retriever(search_kwargs={"k": 3})
        st.success("✅ Ready! Ask your question.")

# ----------------------
# Prompt
# ----------------------
prompt = PromptTemplate(
    template="""
You are a stock market assistant.

Answer ONLY from context.
If not found, say "I don't know".

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

# ----------------------
# Question Section
# ----------------------
query = st.text_input("💬 Ask your question (e.g. 'Is Tata Motors a good buy?')")
ask_btn = st.button("Ask Me")

if ask_btn:
    if not query:
        st.warning("Please enter a question")

    elif "retriever" not in st.session_state:
        st.warning("⚠️ Please submit URLs first")

    else:
        with st.spinner("Analyzing..."):
            docs = st.session_state.retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            chain = prompt | llm
            response = chain.invoke({
                "context": context,
                "question": query
            })

            st.subheader("📊 Answer")
            st.write(response.content)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("Built with ❤️ using Groq + LangChain + Streamlit")
