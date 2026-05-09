import os
import time
import operator
from typing import List, Annotated, TypedDict, Union
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import END, StateGraph, START

# 1. Setup
if "NVIDIA_API_KEY" not in os.environ:
    os.environ["NVIDIA_API_KEY"] = "SET_YOUR_KEY_HERE"

llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)

# 2. State Definition with 'operator.add'
# This is CRITICAL: it tells LangGraph to APPEND new messages to the list
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_step: str

# 3. Define the Supervisor Node
def supervisor_node(state):
    print("\n--- SUPERVISOR: DECIDING NEXT AGENT ---")
    time.sleep(2) # Safety delay to avoid 429
    
    # We look at the last message to see if it's already an answer
    last_message = state["messages"][-1].content
    
    # Simple logic: If an expert has already spoken, we finish
    if "I am the" in last_message:
        print("  - DECISION: FINISH (Task looks complete)")
        return {"next_step": "FINISH"}
    
    prompt = f"""You are a manager. Decide the next step.
    Question: {last_message}
    Respond with exactly one word: SEARCH_EXPERT or GENERAL_RESEARCHER."""
    
    decision = llm.invoke(prompt).content.strip().upper()
    print(f"  - SUPERVISOR DECISION: {decision}")
    
    if "SEARCH" in decision: return {"next_step": "SEARCH_EXPERT"}
    return {"next_step": "GENERAL_RESEARCHER"}

# 4. Define Worker Nodes
def search_expert_node(state):
    print("--- SEARCH EXPERT: WORKING ---")
    time.sleep(1)
    return {"messages": [AIMessage(content="I am the Search Expert. I found info about AI bias in your documents.")]}

def general_researcher_node(state):
    print("--- GENERAL RESEARCHER: WORKING ---")
    time.sleep(1)
    return {"messages": [AIMessage(content="I am the General Researcher. AI ethics is the study of moral implications of AI.")]}

# 5. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("search_expert", search_expert_node)
workflow.add_node("general_researcher", general_researcher_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_step"],
    {
        "SEARCH_EXPERT": "search_expert",
        "GENERAL_RESEARCHER": "general_researcher",
        "FINISH": END
    }
)

workflow.add_edge("search_expert", "supervisor")
workflow.add_edge("general_researcher", "supervisor")

app = workflow.compile()

if __name__ == "__main__":
    print("🏢 Multi-Agent System (V2 - Anti-Loop Fix)\n")
    inputs = {"messages": [HumanMessage(content="Tell me about AI bias.")]}
    
    try:
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Node '{key}' completed.")
    except Exception as e:
        print(f"\n❌ API Error: {e}")
        print("Tip: Wait 30 seconds for the NVIDIA rate limit to reset.")
