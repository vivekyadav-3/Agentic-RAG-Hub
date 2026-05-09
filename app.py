# --- STREAMLIT CLOUD SQLITE3 FIX ---
# Streamlit Cloud has an old SQLite3 version. This overrides it with the newer pysqlite3-binary.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END FIX ---

import streamlit as st
import asyncio
import os
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langgraph.checkpoint.memory import MemorySaver

# --- ARCHITECT LEVEL: ASYNC SUPPORT ---
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# --- LANGSMITH OBSERVABILITY ---
def enable_langsmith():
    if "LANGCHAIN_API_KEY" in os.environ:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "DocIntelligence-Architect"

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_llm(provider, api_key):
    """Factory function to load the correct LLM based on the provider."""
    if provider == "🟢 NVIDIA (Llama 3.1)":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        os.environ["NVIDIA_API_KEY"] = api_key
        return ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
    elif provider == "🟡 OpenAI (GPT-4o)":
        from langchain_openai import ChatOpenAI
        os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(model="gpt-4o", api_key=api_key)
    elif provider == "🔵 Google (Gemini 1.5 Flash)":
        from langchain_google_genai import ChatGoogleGenerativeAI
        os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

st.set_page_config(page_title="DocIntelligence Hub", page_icon="🧠", layout="wide")

# --- SESSION STATE ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = MemorySaver()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Architect Settings")

    # --- PROVIDER SELECTOR ---
    provider = st.selectbox(
        "🤖 AI Model Provider",
        ["🟢 NVIDIA (Llama 3.1)", "🟡 OpenAI (GPT-4o)", "🔵 Google (Gemini 1.5 Flash)"]
    )

    # Dynamic API key label based on provider
    key_labels = {
        "🟢 NVIDIA (Llama 3.1)": ("NVIDIA API Key", "Get free key → build.nvidia.com", "NVIDIA_API_KEY"),
        "🟡 OpenAI (GPT-4o)": ("OpenAI API Key", "Get key → platform.openai.com", "OPENAI_API_KEY"),
        "🔵 Google (Gemini 1.5 Flash)": ("Google API Key", "Get free key → aistudio.google.com", "GOOGLE_API_KEY"),
    }
    label, hint, env_key = key_labels[provider]
    st.caption(f"💡 {hint}")
    api_key = st.text_input(label, type="password", value=os.environ.get(env_key, ""))

    langsmith_key = st.text_input("LangChain API Key (Optional)", type="password")
    if langsmith_key:
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        enable_langsmith()

    st.divider()
    st.header("📂 Knowledge Base")
    chunk_strategy = st.selectbox("Chunking Strategy", ["Standard", "Semantic"])
    search_type = st.selectbox("Search Logic", ["Similarity (Standard)", "MMR (Diverse Results)"])
    enable_multiquery = st.toggle("Enable Multi-Query Strategist (Better Intent)", value=True)
    enable_hybrid = st.toggle("Enable Hybrid Search (Keywords + Meaning)", value=True)
    enable_rerank = st.toggle("Enable AI Reranking (Elite Accuracy)", value=False)
    enable_agentic = st.toggle("Enable Agentic Mode (Async/Stream)", value=True)
    uploaded_files = st.file_uploader("Upload Documents (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

    if st.button("🚀 Process Documents"):
        if api_key and uploaded_files:
            with st.spinner("Processing..."):
                embeddings = load_embeddings()
                all_docs = []
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name
                    loader = PyPDFLoader(tmp_path) if f.name.endswith(".pdf") else TextLoader(tmp_path)
                    all_docs.extend(loader.load())
                    os.unlink(tmp_path)

                splitter = SemanticChunker(embeddings) if chunk_strategy == "Semantic" else RecursiveCharacterTextSplitter(chunk_size=1000)
                chunks = splitter.split_documents(all_docs)
                st.session_state.vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./db_chronos")
                st.success("Brain Updated!")
        elif not api_key:
            st.error("Please enter your API key first!")
        else:
            st.warning("Please upload at least one document.")

# --- MAIN UI ---
st.title("🧠 DocIntelligence Hub")

# Persistence Load
if st.session_state.vector_store is None and os.path.exists("./db_chronos"):
    st.session_state.vector_store = Chroma(persist_directory="./db_chronos", embedding_function=load_embeddings())

if st.session_state.vector_store and api_key:
    # Load the selected LLM
    try:
        llm = load_llm(provider, api_key)
    except Exception as e:
        st.error(f"⚠️ Could not load LLM. Check your API key. Error: {e}")
        st.stop()

    # 1. Base Retriever
    search_kwargs = {"k": 5}
    if search_type == "MMR (Diverse Results)":
        vector_retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    else:
        vector_retriever = st.session_state.vector_store.as_retriever(search_kwargs=search_kwargs)

    # 2. Hybrid Logic
    if enable_hybrid:
        all_docs = st.session_state.vector_store.get()["documents"]
        if all_docs:
            bm25_retriever = BM25Retriever.from_texts(all_docs)
            bm25_retriever.k = 3
            retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.3, 0.7])
        else:
            retriever = vector_retriever
    else:
        retriever = vector_retriever

    # 3. Multi-Query
    if enable_multiquery:
        retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    # 4. Reranker (NVIDIA Only)
    if enable_rerank:
        if "NVIDIA" in provider:
            try:
                from langchain_nvidia_ai_endpoints import NVIDIARerank
                reranker = NVIDIARerank(model="nvidia/llama-3.2-nv-rerank-27b-v1")
                retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
            except Exception:
                st.warning("⚠️ Reranker unavailable. Continuing without it.")
        else:
            st.info("ℹ️ AI Reranking is only available with the NVIDIA provider.")

    # Chains
    ctx_prompt = ChatPromptTemplate.from_messages([
        ("system", "Formulate a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    ctx_chain = ctx_prompt | llm | StrOutputParser()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on context: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = qa_prompt | llm | StrOutputParser()

    # --- RENDER NATIVE MEMORY ---
    config = {"configurable": {"thread_id": "streamlit_user"}}
    history = st.session_state.agent_memory.get(config)
    messages = history.values["messages"] if history else []

    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if enable_agentic:
                from agent_graph import get_agent_graph
                graph_app = get_agent_graph(llm, retriever, ctx_chain, qa_chain, st.session_state.agent_memory)

                response_placeholder = st.empty()
                full_response = [""]

                async def stream_output():
                    async for event in graph_app.astream_events(
                        {"question": prompt, "messages": [HumanMessage(content=prompt)]},
                        config=config, version="v2"
                    ):
                        if event["event"] == "on_chat_model_stream" and "final_node" in event.get("tags", []):
                            content = event["data"]["chunk"].content
                            full_response[0] += content
                            response_placeholder.markdown(full_response[0] + "▌")
                    response_placeholder.markdown(full_response[0])

                run_async(stream_output())
            else:
                with st.spinner("Thinking..."):
                    docs = retriever.invoke(prompt)
                    context = "\n\n".join(d.page_content for d in docs)
                    res = qa_chain.invoke({"context": context, "chat_history": messages, "input": prompt})
                    st.markdown(res)
else:
    st.info("⬅️ Setup your API key and documents in the sidebar to start.")
