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
# 🔐 SET API KEY (hardcoded)
# ----------------------
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="RAG App", layout="wide")
st.title("📊 RAG Assistant")

# ----------------------
# Sidebar URL Form (3 fields)
# ----------------------
st.sidebar.subheader("🌐 Enter URLs")

with st.sidebar.form("url_form"):
    url1 = st.text_input("URL 1")
    url2 = st.text_input("URL 2")
    url3 = st.text_input("URL 3")

    submit_urls = st.form_submit_button("Submit URLs")

# Prepare URL list
url_list = []
if submit_urls:
    url_list = [u for u in [url1, url2, url3] if u]

# ----------------------
# Initialize LLM
# ----------------------
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0
)

# ----------------------
# Build Vector Store
# ----------------------
@st.cache_resource
def build_vector_store(url_list):
    loader = UnstructuredURLLoader(urls=url_list)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(data)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_index = FAISS.from_documents(docs, embedding)

    return vector_index

# ----------------------
# Prompt
# ----------------------
prompt = PromptTemplate(
    template="""
You are a financial assistant.

Answer ONLY from the context.
If answer is not found, say "I don't know".

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

# ----------------------
# RAG Section
# ----------------------
if submit_urls and url_list:

    with st.spinner("Processing URLs..."):
        vector_index = build_vector_store(url_list)
        retriever = vector_index.as_retriever(search_kwargs={"k": 5})

    query = st.text_input("Ask your question:")

    if query:
        with st.spinner("Thinking..."):
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            chain = prompt | llm
            response = chain.invoke({
                "context": context,
                "question": query
            })

            st.subheader("📚 RAG Answer")
            st.write(response.content)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("Built with ❤️ using Groq + LangChain + Streamlit")
