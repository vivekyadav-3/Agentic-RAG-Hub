import os
from typing import List
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 1. Setup API Key
if "NVIDIA_API_KEY" not in os.environ:
    os.environ["NVIDIA_API_KEY"] = "SET_YOUR_KEY_HERE"

llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)

# 2. Query Planner Prompt
# WHY: Complex questions often fail in RAG because one search query isn't enough.
# Decomposition ensures we retrieve context for EVERY part of the question.
planner_prompt = ChatPromptTemplate.from_template(
    """You are an expert search strategist. 
    Break down the following complex user question into 2 or 3 simpler, standalone sub-questions for a RAG system.
    
    User Question: {question}
    
    Return the sub-questions as a JSON list of strings. 
    Example format: ["sub-question 1", "sub-question 2"]
    Do not include any other text, only the JSON list."""
)

planner_chain = planner_prompt | llm | JsonOutputParser()

if __name__ == "__main__":
    print("🎯 Running Query Planner (Lecture 98)...\n")
    
    complex_question = "Compare the 'Black Box' problem and 'Algorithmic Bias' based on the document."
    print(f"Original Question: {complex_question}\n")
    
    try:
        plan = planner_chain.invoke({"question": complex_question})
        print("📋 DECOMPOSED SEARCH PLAN:")
        for i, sub_q in enumerate(plan, 1):
            print(f"  {i}. {sub_q}")
    except Exception as e:
        print(f"Error during decomposition: {e}")
