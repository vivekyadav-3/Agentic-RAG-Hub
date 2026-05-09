import asyncio
import operator
from typing import List, Annotated, TypedDict, Any, Union
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# --- 1. STATE DEFINITION (Pure Native Memory) ---
class GraphState(TypedDict):
    # This automatically appends new messages, creating a permanent history in the checkpointer
    messages: Annotated[List[BaseMessage], operator.add]
    question: str
    generation: str
    documents: List[Any]

# --- 2. ASYNC RETRY WRAPPER ---
async def with_retry_async(func, retries=3, backoff=2):
    for i in range(retries):
        try:
            return await func()
        except Exception as e:
            if i == retries - 1:
                print(f"Failed after {retries} attempts: {e}")
                raise e
            await asyncio.sleep(backoff * (i + 1))

# --- 3. GRAPH ENGINE (Async) ---
def get_agent_graph(llm, retriever, contextualize_q_chain, final_qa_chain, memory):
    
    async def retrieve_node(state: GraphState):
        try:
            # We use the built-in 'messages' history for context
            standalone_q = await contextualize_q_chain.ainvoke({
                "input": state["question"], 
                "chat_history": state["messages"]
            })
            docs = await with_retry_async(lambda: retriever.ainvoke(standalone_q))
        except Exception as e:
            print(f"Retrieval error: {e}")
            docs = []
        return {"documents": docs, "question": state["question"]}

    async def grade_node(state: GraphState):
        prompt_grade = ChatPromptTemplate.from_template(
            "Grade the relevance of this doc to the question: {question}. Doc: {document}. Answer with exactly one word: 'yes' or 'no'."
        )
        grader = prompt_grade | llm | StrOutputParser()
        
        async def grade_single_doc(d):
            try:
                res = await with_retry_async(
                    lambda: grader.ainvoke({"question": state["question"], "document": d.page_content}), 
                    retries=2, backoff=1
                )
                return d if "yes" in res.lower() else None
            except Exception as e:
                print(f"Grading error: {e}")
                return d # Keep it if there's an API failure just to be safe
                
        # Fire off all grading tasks at the exact same time (Parallel)
        tasks = [grade_single_doc(d) for d in state["documents"][:5]]
        results = await asyncio.gather(*tasks)
        
        # Filter out the documents that got a 'no' (which return None)
        relevant_docs = [doc for doc in results if doc is not None]
        
        return {"documents": relevant_docs}

    async def generate_node(state: GraphState):
        context = "\n\n".join(d.page_content for d in state["documents"])
        try:
            # The final chain is used to generate. 
            # Note: We stream tokens in app.py using astream_events
            ans = await with_retry_async(lambda: final_qa_chain.with_config({"tags": ["final_node"]}).ainvoke({
                "context": context,
                "chat_history": state["messages"],
                "input": state["question"]
            }))
        except Exception as e:
            print(f"Generation error: {e}")
            ans = "Error generating answer."
        return {"generation": ans}

    async def transform_node(state: GraphState):
        prompt_trans = ChatPromptTemplate.from_template(
            "Rewrite the question to be better for vector search. Output ONLY the question. Original: {question}"
        )
        trans_chain = prompt_trans | llm | StrOutputParser()
        try:
            new_q = await with_retry_async(lambda: trans_chain.ainvoke({"question": state["question"]}))
        except Exception as e:
            new_q = state["question"]
        return {"question": new_q}

    # --- BUILD GRAPH ---
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
    
    return workflow.compile(checkpointer=memory)
