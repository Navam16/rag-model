
# ================================
# app.py — LangGraph RAG Chatbot
# ================================

import os
import uuid
import tempfile
import streamlit as st
from typing import TypedDict, Annotated, Optional, Dict, Any

# ---------------- ENV ----------------
# Gemini key should be stored in:
# Streamlit → Settings → Secrets
# or Colab env variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# ---------------- LANGCHAIN / LANGGRAPH ----------------
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.embeddings import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI


# =========================
# LLM & EMBEDDINGS
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# VECTOR STORAGE (PER THREAD)
# =========================
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        temp_path = f.name

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    _THREAD_RETRIEVERS[thread_id] = retriever
    _THREAD_METADATA[thread_id] = {
        "filename": filename,
        "documents": len(docs),
        "chunks": len(chunks),
    }

    os.remove(temp_path)
    return _THREAD_METADATA[thread_id]


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(thread_id, {})


def _get_retriever(thread_id: Optional[str]):
    return _THREAD_RETRIEVERS.get(thread_id)


# =========================
# TOOLS
# =========================
search_tool = DuckDuckGoSearchRun()


@tool
def rag_tool(query: str, thread_id: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF document.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document uploaded for this chat."}

    docs = retriever.invoke(query)
    return {
        "query": query,
        "context": [d.page_content for d in docs],
    }


tools = [search_tool, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# =========================
# LANGGRAPH
# =========================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config=None):
    system = SystemMessage(
        content=(
            "You are a helpful AI assistant. "
            "If the question is about the uploaded PDF, "
            "use rag_tool with the correct thread_id."
        )
    )

    response = llm_with_tools.invoke(
        [system, *state["messages"]],
        config=config,
    )
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

chatbot = graph.compile(checkpointer=MemorySaver())


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="LangGraph RAG Chatbot", layout="wide")


def new_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []


if "thread_id" not in st.session_state:
    new_chat()

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------- Sidebar --------
st.sidebar.title("📄 PDF Chatbot")
st.sidebar.markdown(f"**Thread ID**  `{st.session_state.thread_id}`")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    new_chat()
    st.rerun()

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    with st.sidebar.status("Indexing PDF...", expanded=True):
        ingest_pdf(
            uploaded_pdf.getvalue(),
            st.session_state.thread_id,
            uploaded_pdf.name,
        )
    st.sidebar.success("PDF indexed successfully")

meta = thread_document_metadata(st.session_state.thread_id)
if meta:
    st.sidebar.caption(
        f"📘 {meta['filename']} | Pages: {meta['documents']} | Chunks: {meta['chunks']}"
    )


# -------- Main Chat --------
st.title("🤖 Multi-Utility RAG Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about the document or anything else...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        full_response = ""

        for message, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": st.session_state.thread_id}},
            stream_mode="messages",
        ):
            if isinstance(message, ToolMessage):
                st.status(f"🔧 Using `{message.name}`", expanded=False)

            if isinstance(message, AIMessage) and isinstance(message.content, str): full_response += message.content
                st.write(full_response)

                     
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

st.divider()
st.caption("⚡ Powered by LangGraph + Gemini + HuggingFace Embeddings")
