import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIARerank
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers import JsonOutputParser
from typing import List, TypedDict
import os
import tempfile

# Cache the embedding model so it doesn't reload every time
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocIntelligence Hub", page_icon="🧠", layout="wide")

# Simplified CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("NVIDIA API Key", type="password", value=os.environ.get("NVIDIA_API_KEY", ""))
    
    st.divider()
    
    st.header("📂 Knowledge Base")
    chunk_strategy = st.selectbox("Chunking Strategy", ["Standard (Fast)", "Semantic (Smart / AI-based)"])
    search_type = st.selectbox("Search Logic", ["Similarity (Standard)", "MMR (Diverse Results)"])
    enable_multiquery = st.toggle("Enable Multi-Query Strategist (Better Intent)", value=True)
    enable_hybrid = st.toggle("Enable Hybrid Search (Keywords + Meaning)", value=True)
    enable_rerank = st.toggle("Enable AI Reranking (Elite Accuracy)", value=False)
    enable_agentic = st.toggle("Enable Agentic Mode (Self-Correction & Grading)", value=True)
    uploaded_files = st.file_uploader("Upload Documents (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("🚀 Process Documents"):
        if not api_key:
            st.error("Please provide an NVIDIA API Key!")
        elif uploaded_files:
            with st.spinner("Analyzing documents..."):
                os.environ["NVIDIA_API_KEY"] = api_key
                embeddings = load_embeddings()
                
                all_docs = []
                for uploaded_file in uploaded_files:
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load based on type
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path)
                    
                    loaded_docs = loader.load()
                    # Add filename to metadata for citations
                    for d in loaded_docs:
                        d.metadata["source"] = uploaded_file.name
                    
                    all_docs.extend(loaded_docs)
                    os.unlink(tmp_path)
                
                # Choose Chunking Strategy
                if chunk_strategy == "Semantic (Smart / AI-based)":
                    text_splitter = SemanticChunker(embeddings)
                    st.info("Using AI to find logical breakpoints in text...")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                
                chunks = text_splitter.split_documents(all_docs)
                
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=os.path.abspath("./db")
                )
                st.success(f"Brain Updated! {len(all_docs)} files added.")
        else:
            st.warning("No files uploaded.")

# --- MAIN INTERFACE ---
st.title("🧠 DocIntelligence Hub")
st.subheader("Interactive Multi-Document RAG Agent")

# Load existing vector store if available
if st.session_state.vector_store is None:
    if os.path.exists(os.path.abspath("./db")):
        embeddings = load_embeddings()
        st.session_state.vector_store = Chroma(persist_directory=os.path.abspath("./db"), embedding_function=embeddings)

# Setup Chains
if st.session_state.vector_store and api_key:
    os.environ["NVIDIA_API_KEY"] = api_key
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
    
    # 1. Base Retriever (Vector)
    search_kwargs = {"k": 10}
    if search_type == "MMR (Diverse Results)":
        vector_retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    else:
        vector_retriever = st.session_state.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    # 2. Hybrid Logic (BM25 + Vector)
    if enable_hybrid:
        # We need the actual document objects for BM25
        # Since Chroma is persistent, we can get all documents
        all_docs = st.session_state.vector_store.get()["documents"]
        if all_docs:
            bm25_retriever = BM25Retriever.from_texts(all_docs)
            bm25_retriever.k = 5
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever], 
                weights=[0.3, 0.7]
            )
        else:
            retriever = vector_retriever
    else:
        retriever = vector_retriever
    
    # 3. Multi-Query Strategy (The Strategist)
    if enable_multiquery:
        retriever = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=llm
        )
    
    # 4. Setup Reranker (with safety catch)
    if enable_rerank:
        try:
            # Using the newest stable reranker model
            reranker = NVIDIARerank(model="nvidia/llama-3.2-nv-rerank-27b-v1")
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=retriever
            )
        except Exception:
            st.warning("⚠️ Reranker model is temporarily unavailable. Falling back to standard search.")
            compression_retriever = retriever
    else:
        compression_retriever = retriever

    # Prompts
    contextualize_q_system_prompt = "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history."
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    qa_system_prompt = """You are a helpful AI assistant. Use the following context to answer the question concisely.
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    final_qa_chain = qa_prompt | llm | StrOutputParser()

    # Display Chat
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # User Input
    if prompt := st.chat_input("Ask about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if enable_agentic:
                # --- AGENTIC RAG (LANGGRAPH) ---
                class GraphState(TypedDict):
                    question: str
                    generation: str
                    documents: List[str]

                def retrieve_node(state):
                    with st.status("🔍 Retrieving relevant context..."):
                        import time
                        try:
                            standalone_q = contextualize_q_chain.invoke({"input": state["question"], "chat_history": st.session_state.chat_history})
                            time.sleep(2) # Prevent rapid API calls
                            docs = compression_retriever.invoke(standalone_q)
                        except Exception as e:
                            print(f"Retrieval error: {e}")
                            # Fallback to standard vector search if complex retrieval fails due to rate limits
                            standalone_q = state["question"]
                            docs = st.session_state.vector_store.as_retriever().invoke(standalone_q)
                        return {"documents": docs, "question": state["question"]}

                def grade_node(state):
                    with st.status("⚖️ Grading document relevance (Top 3 for speed)..."):
                        import time
                        prompt_grade = ChatPromptTemplate.from_template(
                            "Grade the relevance of this doc to the question: {question}. Doc: {document}. Answer with exactly one word: 'yes' or 'no'."
                        )
                        grader = prompt_grade | llm | StrOutputParser()
                        relevant_docs = []
                        # Only grade the top 3 documents to save massive time
                        for d in state["documents"][:3]:
                            try:
                                res = grader.invoke({"question": state["question"], "document": d.page_content})
                                if "yes" in res.lower():
                                    relevant_docs.append(d)
                                time.sleep(0.5) # Reduced sleep since we only do 3 calls
                            except Exception as e:
                                print(f"Grading error: {e}")
                                relevant_docs.append(d)
                                time.sleep(1)
                        return {"documents": relevant_docs, "question": state["question"]}

                def generate_node(state):
                    with st.status("✍️ Generating final answer..."):
                        import time
                        context = "\n\n".join(d.page_content for d in state["documents"])
                        try:
                            time.sleep(2)
                            ans = final_qa_chain.invoke({
                                "context": context,
                                "chat_history": st.session_state.chat_history,
                                "input": state["question"]
                            })
                        except Exception as e:
                            print(f"Rate limit hit in generate_node. Waiting 5s... {e}")
                            time.sleep(5)
                            ans = final_qa_chain.invoke({
                                "context": context,
                                "chat_history": st.session_state.chat_history,
                                "input": state["question"]
                            })
                        return {"generation": ans, "documents": state["documents"]}

                def transform_node(state):
                    with st.status("🔄 No relevant info found. Transforming query..."):
                        import time
                        prompt_trans = ChatPromptTemplate.from_template(
                            "You are a strict search query optimizer. Rewrite the following question to be better for a vector database search. Output ONLY the rewritten question string, with NO introductory text, NO quotes, and NO conversational filler.\n\nOriginal Question: {question}"
                        )
                        trans_chain = prompt_trans | llm | StrOutputParser()
                        try:
                            time.sleep(2)
                            new_q = trans_chain.invoke({"question": state["question"]})
                        except Exception as e:
                            print(f"Rate limit hit in transform_node. Waiting 5s... {e}")
                            time.sleep(5)
                            new_q = trans_chain.invoke({"question": state["question"]})
                        return {"question": new_q, "documents": state["documents"]}

                # Build Graph
                workflow = StateGraph(GraphState)
                workflow.add_node("retrieve", retrieve_node)
                workflow.add_node("grade", grade_node)
                workflow.add_node("generate", generate_node)
                workflow.add_node("transform", transform_node)

                workflow.add_edge(START, "retrieve")
                workflow.add_edge("retrieve", "grade")
                workflow.add_conditional_edges(
                    "grade",
                    lambda x: "generate" if x["documents"] else "transform",
                    {"generate": "generate", "transform": "transform"}
                )
                workflow.add_edge("transform", "retrieve")
                workflow.add_edge("generate", END)
                
                graph_app = workflow.compile()
                
                # Execute with a recursion limit so it doesn't loop forever on missing info
                docs = []
                try:
                    final_state = graph_app.invoke({"question": prompt}, config={"recursion_limit": 5})
                    answer = final_state["generation"]
                    docs = final_state.get("documents", [])
                    st.markdown(answer)
                except Exception as e:
                    # If it exceeds recursion limit, it means it couldn't find relevant info after multiple tries
                    answer = "I searched the documents multiple times but could not find any relevant information to answer your question."
                    st.warning(answer)
            else:
                # --- STANDARD RAG ---
                with st.spinner("Thinking..."):
                    standalone_q = contextualize_q_chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
                    docs = compression_retriever.invoke(standalone_q)
                    context = "\n\n".join(doc.page_content for doc in docs)
                    answer = final_qa_chain.invoke({
                        "context": context,
                        "chat_history": st.session_state.chat_history,
                        "input": standalone_q
                    })
                    st.markdown(answer)
            
            # Sources
            if docs:
                sources = list(set([f"{os.path.basename(doc.metadata.get('source', 'unknown'))} (Page {doc.metadata.get('page', 'N/A')})" for doc in docs]))
                with st.expander("📚 View Sources"):
                    for s in sources:
                        st.write(f"- {s}")
        
        # Save History
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=answer))

else:
    st.info("👋 Welcome! Please provide your NVIDIA API Key and upload documents in the sidebar to start chatting.")
