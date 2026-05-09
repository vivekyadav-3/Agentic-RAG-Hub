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

# 2. State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_node: str

# 3. CEO Node (Top Level Supervisor)
def ceo_node(state):
    print("\n🏢 CEO: ANALYZING CORPORATE STRATEGY...")
    time.sleep(1)
    last_msg = state["messages"][-1].content
    
    if "FINAL ANSWER" in last_msg:
        return {"next_node": "FINISH"}
    
    prompt = f"CEO Decision. Question: {last_msg}. Route to: PRODUCT_DEPT or RESEARCH_DEPT?"
    decision = llm.invoke(prompt).content.strip().upper()
    print(f"  - CEO DECISION: {decision}")
    
    if "PRODUCT" in decision: return {"next_node": "PRODUCT_DEPT"}
    return {"next_node": "RESEARCH_DEPT"}

# 4. Department Nodes (Sub-Supervisors)
def product_dept_node(state):
    print("📦 PRODUCT DEPT: PROCESSING PRODUCT SEARCH...")
    return {"messages": [AIMessage(content="[Product Dept] I have searched the documents and found the answer. FINAL ANSWER: The 'Black Box' problem is about lack of transparency.")]}

def research_dept_node(state):
    print("🔬 RESEARCH DEPT: CONDUCTING DEEP RESEARCH...")
    return {"messages": [AIMessage(content="[Research Dept] I have analyzed the definitions. FINAL ANSWER: AI Ethics refers to moral principles.")]}

# 5. Build the Hierarchy
workflow = StateGraph(AgentState)

workflow.add_node("ceo", ceo_node)
workflow.add_node("product_dept", product_dept_node)
workflow.add_node("research_dept", research_dept_node)

workflow.add_edge(START, "ceo")

workflow.add_conditional_edges(
    "ceo",
    lambda x: x["next_node"],
    {
        "PRODUCT_DEPT": "product_dept",
        "RESEARCH_DEPT": "research_dept",
        "FINISH": END
    }
)

# Departments report back to CEO
workflow.add_edge("product_dept", "ceo")
workflow.add_edge("research_dept", "ceo")

app = workflow.compile()

if __name__ == "__main__":
    print("🚀 Hierarchical Multi-Agent System STARTING...\n")
    inputs = {"messages": [HumanMessage(content="Explain the Black Box problem.")]}
    
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}' execution finished.")
