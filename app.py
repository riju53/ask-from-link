# app.py
import os
import streamlit as st

# LangChain + Groq
from langchain-groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Stock API
import yfinance as yf

# ----------------------
# Load environment
# ----------------------
# Set API key
os.environ["GROQ_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="RAG + Stock App", layout="wide")
st.title("📊 RAG + Live Stock Assistant")

# Sidebar
st.sidebar.header("Settings")
urls = st.sidebar.text_area("Enter URLs (one per line)", height=150)
stock_symbol = st.sidebar.text_input("Stock Symbol (e.g. TATAMOTORS.NS)")

# ----------------------
# Initialize LLM
# ----------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
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
# Main Logic
# ----------------------
if urls:
    url_list = [u.strip() for u in urls.split("\n") if u.strip()]
    vector_index = build_vector_store(url_list)
    retriever = vector_index.as_retriever(search_kwargs={"k": 5})

    query = st.text_input("Ask your question:")

    if query:
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
# Live Stock Section
# ----------------------
st.divider()
st.subheader("📈 Live Stock Price")

if stock_symbol:
    try:
        stock = yf.Ticker(stock_symbol)
        price = stock.history(period="1d")["Close"].iloc[-1]

        st.success(f"Current Price of {stock_symbol}: ₹ {price:.2f}")
    except Exception as e:
        st.error("Failed to fetch stock data")
