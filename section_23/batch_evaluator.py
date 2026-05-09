import os
import time
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Setup
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)

# 2. Test Dataset (Lecture 117)
# In a real project, you would have 50+ questions here.
test_dataset = [
    {
        "question": "What is the 'Black Box' problem?",
        "context": "The 'Black Box' problem refers to the lack of transparency in AI models.",
        "expected": "It is a lack of transparency in how AI makes decisions."
    },
    {
        "question": "What is Algorithmic Bias?",
        "context": "Algorithmic bias occurs when an AI system produces results that are systematically prejudiced.",
        "expected": "Prejudiced or unfair results produced by an AI system."
    },
    {
        "question": "How long does the MVP roadmap take?",
        "context": "The Rapid AI MVP Delivery Roadmap is a 20-day process.",
        "expected": "20 days."
    }
]

# 3. Judge Logic
def get_faithfulness_score(q, c, a):
    prompt = ChatPromptTemplate.from_template(
        "Rate the faithfulness of this answer (0.0 to 1.0) based ONLY on the context. Answer only with the number. \nContext: {c}\nAnswer: {a}"
    )
    chain = prompt | llm | StrOutputParser()
    try:
        return float(chain.invoke({"c": c, "a": a}).strip())
    except:
        return 0.0

# 4. Run Batch Evaluation (Lecture 118)
if __name__ == "__main__":
    print("📋 STARTING BATCH EVALUATION...\n")
    total_score = 0
    
    for i, test in enumerate(test_dataset, 1):
        print(f"Testing Question {i}: {test['question']}")
        
        # In a real run, you'd call your RAG here. 
        # For this demo, we'll simulate the RAG's response.
        actual_answer = "This is a placeholder for the RAG response." # Simulating RAG
        
        score = get_faithfulness_score(test['question'], test['context'], test['expected'])
        total_score += score
        print(f"  - Score: {score}")
        time.sleep(1) # Safety delay
    
    final_avg = (total_score / len(test_dataset)) * 100
    print(f"\n✨ FINAL RAG REPORT CARD: {final_avg:.1f}% ACCURACY")
