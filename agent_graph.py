import os
import warnings
from typing import List, TypedDict
from typing_extensions import Annotated

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph, START

warnings.filterwarnings("ignore")

# --- 1. SETUP KNOWLEDGE BASE ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Use the same directory as the app.py to share data
persist_directory = "./db"
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- 2. DEFINE STATE ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[str]

# --- 3. INITIALIZE LLM ---
# Fallback API Key from main.py if not in environment
# Do not hardcode API keys. Use environment variables or sidebar input.
if "NVIDIA_API_KEY" not in os.environ:
    print("Please set the NVIDIA_API_KEY environment variable.")

print("Initializing LLM and Embeddings...")
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)

# --- 4. DEFINE NODES ---

def retrieve(state):
    """
    Retrieve documents from vectorstore
    """
    print("\n--- RETRIEVING DOCUMENTS ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": [d.page_content for d in documents], "question": question}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    import time
    print("\n--- CHECKING DOCUMENT RELEVANCE ---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_template(
        "Grade the relevance of this doc to the question: {question}. Doc: {document}. Answer with exactly one word: 'yes' or 'no'."
    )
    grader_chain = prompt | llm | StrOutputParser()
    
    filtered_docs = []
    # Only grade top 3 for speed and API safety
    for d in documents[:3]:
        try:
            res = grader_chain.invoke({"question": question, "document": d})
            if "yes" in res.lower():
                print("  - GRADE: DOCUMENT RELEVANT")
                filtered_docs.append(d)
            else:
                print("  - GRADE: DOCUMENT NOT RELEVANT")
            time.sleep(1) # Rate limit safety
        except Exception as e:
            print(f"  - GRADING ERROR: {e}")
            filtered_docs.append(d)
    
    return {"documents": filtered_docs, "question": question}

def generate(state):
    """
    Generate answer
    """
    print("\n--- GENERATING ANSWER ---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \n
        If you don't know the answer, just say that you don't know. \n
        Question: {question} \n
        Context: {context} \n
        Answer:"""
    )
    
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(docs)
    
    rag_chain = prompt | llm | StrOutputParser()
    
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def transform_query(state):
    """
    Transform the query to produce a better question.
    """
    import time
    print("\n--- TRANSFORMING QUERY ---")
    question = state["question"]
    
    prompt = ChatPromptTemplate.from_template(
        "You are a strict search query optimizer. Rewrite the following question to be better for a vector database search. Output ONLY the rewritten question string, with NO introductory text, NO quotes, and NO conversational filler.\n\nOriginal Question: {question}"
    )
    
    chain = prompt | llm | StrOutputParser()
    time.sleep(2) # Prevent rapid API calls
    better_question = chain.invoke({"question": question})
    print(f"  - IMPROVED QUESTION: {better_question}")
    return {"documents": state["documents"], "question": better_question}

# --- 5. DEFINE CONDITIONAL EDGES ---

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    print("\n--- DECIDING NEXT STEP ---")
    if not state["documents"]:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("  - DECISION: TRANSFORM QUERY")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("  - DECISION: GENERATE ANSWER")
        return "generate"

# --- 6. BUILD GRAPH ---
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve) 
workflow.add_node("grade_documents", grade_documents) 
workflow.add_node("generate", generate) 
workflow.add_node("transform_query", transform_query) 

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# --- 7. RUN THE AGENT ---
if __name__ == "__main__":
    print("\n--- DETAILED AGENTIC RAG STARTING ---")
    
    # Test Question
    inputs = {"question": "What is the 'Black Box' problem in AI ethics?"}
    
    # Run with recursion limit
    try:
        for output in app.stream(inputs, config={"recursion_limit": 5}):
            for key, value in output.items():
                print(f"Node '{key}':")
    except Exception as e:
        print(f"\n--- STOPPED: {e} ---")
        value = {"generation": "I searched multiple times but could not find relevant info."}
    
    print("\n--- FINAL GENERATION ---")
    print(value.get("generation", "No generation produced."))
