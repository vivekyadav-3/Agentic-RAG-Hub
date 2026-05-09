import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Setup LLM and API Key (Make sure to set this in your environment)
if "NVIDIA_API_KEY" not in os.environ:
    os.environ["NVIDIA_API_KEY"] = "SET_YOUR_KEY_HERE"

llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Setup a quick Vector Store for RAG
# We'll use the existing Chroma DB from your previous project to save time
persist_directory = "./db"  # Changed from ../db to ./db
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 3. Chain of Thought Prompt for RAG
# WHY WE DO THIS: We force the LLM to write out its thinking process BEFORE giving the final answer.
# This prevents hallucinations and makes the LLM analyze the context much better.
cot_prompt = ChatPromptTemplate.from_template(
    """You are an intelligent assistant. You have been provided with retrieved context to answer a question.
    
    Before giving your final answer, you MUST think step-by-step. 
    Write your internal reasoning process inside <thought> tags. 
    Analyze the context, decide if it contains the answer, and plan your response.
    After the <thought> block, provide your final, concise answer.
    
    Context:
    {context}
    
    Question: {question}
    """
)

# 4. Build the RAG Chain
# We pass the question to the retriever, format the context, and send it to the prompt.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | cot_prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print("🤖 Running Chain of Thought RAG...\n")
    
    test_question = "What is the 'Black Box' problem in AI ethics?"
    print(f"Question: {test_question}\n")
    
    print("Generating Answer (Notice the <thought> process)...\n")
    response = rag_chain.invoke(test_question)
    
    print(response)
