import os
import time
from typing import List, TypedDict
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START

# 1. Setup
if "NVIDIA_API_KEY" not in os.environ:
    os.environ["NVIDIA_API_KEY"] = "SET_YOUR_KEY_HERE"

llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 2. State Definition
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    iteration: int # Track how many times we've tried

# 3. Nodes

def retrieve_node(state):
    print("--- RETRIEVING ---")
    question = state["question"]
    docs = retriever.invoke(question)
    return {"documents": [d.page_content for d in docs], "question": question}

def generate_node(state):
    print("--- GENERATING ---")
    question = state["question"]
    context = "\n\n".join(state["documents"])
    
    prompt = ChatPromptTemplate.from_template(
        "Use the context to answer the question. Context: {context} \n Question: {question}"
    )
    chain = prompt | llm | StrOutputParser()
    res = chain.invoke({"context": context, "question": question})
    return {"generation": res, "iteration": state.get("iteration", 0) + 1}

def reflection_node(state):
    print("--- REFLECTING (CRITIC) ---")
    # This node critiques the generation
    question = state["question"]
    generation = state["generation"]
    
    prompt = ChatPromptTemplate.from_template(
        """You are a strict grader. Grade the following AI response for faithfulness to the context and accuracy.
        AI Response: {generation}
        User Question: {question}
        
        If the response fully answers the question accurately based on facts, answer with 'satisfactory'.
        If the response is vague, incorrect, or missing details, answer with 'unsatisfactory'.
        Answer with exactly one word: 'satisfactory' or 'unsatisfactory'."""
    )
    critic = prompt | llm | StrOutputParser()
    grade = critic.invoke({"generation": generation, "question": question})
    print(f"  - CRITIC GRADE: {grade}")
    return {"generation": generation, "documents": state["documents"]} # We pass the grade via conditional edge

# 4. Conditional Edge Logic
def decide_to_finish(state):
    # Here we perform the check based on the LLM's feedback
    # (Since we didn't store the grade in state, we re-run the critic or check the generation)
    # For simplicity in this demo, let's assume we store the grade or just re-check it
    prompt = ChatPromptTemplate.from_template(
        "Is this response 'satisfactory' or 'unsatisfactory'? Response: {generation}. Answer only one word."
    )
    grade = (prompt | llm | StrOutputParser()).invoke({"generation": state["generation"]})
    
    if "satisfactory" in grade.lower() or state["iteration"] >= 2:
        print("  - DECISION: FINISH")
        return "finish"
    else:
        print("  - DECISION: RE-GENERATE")
        return "re-generate"

# 5. Build Graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("reflect", reflection_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "reflect")

workflow.add_conditional_edges(
    "reflect",
    decide_to_finish,
    {
        "finish": END,
        "re-generate": "generate"
    }
)

app = workflow.compile()

if __name__ == "__main__":
    inputs = {"question": "What is the 'Black Box' problem?", "iteration": 0}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}' completed.")
    
    print("\n--- FINAL ANSWER ---")
    print(value.get("generation"))
