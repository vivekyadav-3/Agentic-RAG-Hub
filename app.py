import streamlit as st
import asyncio
import os
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank
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
# This allows Streamlit to run async code without blocking
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# --- LANGSMITH OBSERVABILITY ---
# To enable, add your LANGCHAIN_API_KEY to secrets or sidebar
def enable_langsmith():
    if "LANGCHAIN_API_KEY" in os.environ:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "DocIntelligence-Architect"

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.set_page_config(page_title="DocIntelligence Hub", page_icon="🧠", layout="wide")

# --- SESSION STATE ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = MemorySaver()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Architect Settings")
    api_key = st.text_input("NVIDIA API Key", type="password", value=os.environ.get("NVIDIA_API_KEY", ""))
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
                os.environ["NVIDIA_API_KEY"] = api_key
                embeddings = load_embeddings()
                all_docs = []
                for f in uploaded_files:
                    # Create the temp file and ensure it is fully written/closed before loading
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name
                        
                    loader = PyPDFLoader(tmp_path) if f.name.endswith(".pdf") else TextLoader(tmp_path)
                    all_docs.extend(loader.load())
                    os.unlink(tmp_path) # Clean up the temp file
                
                splitter = SemanticChunker(embeddings) if chunk_strategy == "Semantic" else RecursiveCharacterTextSplitter(chunk_size=1000)
                chunks = splitter.split_documents(all_docs)
                st.session_state.vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./db_chronos")
                st.success("Brain Updated!")

# --- MAIN UI ---
st.title("🧠 DocIntelligence Hub")

# Persistence Load
if st.session_state.vector_store is None and os.path.exists("./db_chronos"):
    st.session_state.vector_store = Chroma(persist_directory="./db_chronos", embedding_function=load_embeddings())

if st.session_state.vector_store and api_key:
    os.environ["NVIDIA_API_KEY"] = api_key
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
    
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
        
    # 4. Reranker
    if enable_rerank:
        try:
            reranker = NVIDIARerank(model="nvidia/llama-3.2-nv-rerank-27b-v1")
            retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
        except Exception:
            st.warning("⚠️ Reranker model unavailable.")
    
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
    # We fetch history from the LangGraph Checkpointer
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
                
                # --- ASYNC STREAMING ---
                response_placeholder = st.empty()
                full_response = [""]
                
                async def stream_output():
                    # We use astream_events to capture tokens from the 'generate' node
                    async for event in graph_app.astream_events(
                        {"question": prompt, "messages": [HumanMessage(content=prompt)]},
                        config=config, version="v2"
                    ):
                        # Only stream tokens from the actual final generation step, ignore grading/query processing
                        if event["event"] == "on_chat_model_stream" and "final_node" in event.get("tags", []):
                            content = event["data"]["chunk"].content
                            full_response[0] += content
                            response_placeholder.markdown(full_response[0] + "▌")
                    response_placeholder.markdown(full_response[0])

                run_async(stream_output())
            else:
                # Standard RAG Fallback
                with st.spinner("Thinking..."):
                    docs = retriever.invoke(prompt)
                    context = "\n\n".join(d.page_content for d in docs)
                    res = qa_chain.invoke({"context": context, "chat_history": messages, "input": prompt})
                    st.markdown(res)
else:
    st.info("Setup your API key and documents to start.")
