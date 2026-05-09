import os
import time
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Setup
if "NVIDIA_API_KEY" not in os.environ:
    os.environ["NVIDIA_API_KEY"] = "SET_YOUR_KEY_HERE"

llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)

# 2. The Evaluation Framework
# Instead of installing huge RAGAS libraries for a simple test, 
# we will build a 'Judge LLM' that implements the Faithfulness metric.

def evaluate_faithfulness(question, context, answer):
    print("\n🧐 JUDGE: EVALUATING FAITHFULNESS...")
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert evaluator. Your task is to judge if an AI's answer is faithful to the provided context.
        
        CONTEXT: {context}
        QUESTION: {question}
        AI ANSWER: {answer}
        
        Is every part of the AI ANSWER supported by the CONTEXT? 
        If the AI added information NOT in the context, it is 'NOT FAITHFUL'.
        If it only used provided facts, it is 'FAITHFUL'.
        
        Give a score from 0.0 (Hallucination) to 1.0 (Perfectly Faithful).
        Answer ONLY with the numerical score (e.g., 0.9)."""
    )
    
    judge_chain = prompt | llm | StrOutputParser()
    score = judge_chain.invoke({"question": question, "context": context, "answer": answer})
    return score

if __name__ == "__main__":
    print("📊 RAG Evaluation Dashboard (Lecture 115)\n")
    
    # CASE 1: A Faithful Answer
    q1 = "What is the Black Box problem?"
    c1 = "The Black Box problem refers to the lack of transparency in AI models."
    a1 = "The Black Box problem is about a lack of transparency in AI."
    
    score1 = evaluate_faithfulness(q1, c1, a1)
    print(f"✅ CASE 1 (Good Answer) Score: {score1}")
    
    # CASE 2: A Hallucination
    q2 = "What is the Black Box problem?"
    c2 = "The Black Box problem refers to the lack of transparency in AI models."
    a2 = "The Black Box problem is about lack of transparency and it was first discovered in 1950 by NASA."
    
    score2 = evaluate_faithfulness(q2, c2, a2)
    print(f"❌ CASE 2 (Hallucinated Fact) Score: {score2}")
