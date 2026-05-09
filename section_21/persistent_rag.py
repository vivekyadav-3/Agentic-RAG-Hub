import os
import time
import operator
from typing import List, Annotated, TypedDict

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

# 1. Setup
if "NVIDIA_API_KEY" not in os.environ:
    os.environ["NVIDIA_API_KEY"] = "SET_YOUR_KEY_HERE"

llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)

# 2. State
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 3. Chat Node
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 4. Build Graph with Memory
workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# --- IN-MEMORY CHECKPOINTER ---
# This works the same as Sqlite but doesn't require extra pip installs
memory = MemorySaver()

app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    # We must provide a thread_id to access the memory
    config = {"configurable": {"thread_id": "1"}}
    
    print("🧠 AI Memory Test (Lecture 110)\n")
    
    # Step 1: Introduce yourself
    print("--- STEP 1: Introduction ---")
    input1 = {"messages": [HumanMessage(content="Hi! I am Vivek. Remember my name.")]}
    for event in app.stream(input1, config):
        for value in event.values():
            print(f"AI: {value['messages'][-1].content}")
            
    # Step 2: Test the memory
    print("\n--- STEP 2: The Memory Check ---")
    input2 = {"messages": [HumanMessage(content="Do you remember who I am?")]}
    for event in app.stream(input2, config):
        for value in event.values():
            print(f"AI: {value['messages'][-1].content}")
