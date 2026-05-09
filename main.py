from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os

# 1. Setup
# Do not hardcode API keys.
if "NVIDIA_API_KEY" not in os.environ:
    os.environ["NVIDIA_API_KEY"] = "SET_YOUR_KEY_HERE"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./db", embedding_function=embeddings)
retriever = vectordb.as_retriever()
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

# 2. Prompts
# This prompt helps the AI understand follow-up questions
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# The main prompt for answering
qa_system_prompt = """You are a helpful AI assistant. Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 3. Logic
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# This chain "cleans" the question based on history
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# This chain generates the final answer
# 3. Logic
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the final chain
final_qa_chain = qa_prompt | llm | StrOutputParser()

def get_response(user_query, history):
    # a. Contextualize the question (make it standalone)
    standalone_q = contextualize_q_chain.invoke({"input": user_query, "chat_history": history})
    
    # b. Retrieve docs based on the standalone question
    docs = retriever.invoke(standalone_q)
    context = format_docs(docs)
    
    # c. Generate final answer
    answer = final_qa_chain.invoke({
        "context": context, 
        "chat_history": history, 
        "input": standalone_q
    })
    
    # d. Extract unique sources (filename + page)
    sources = []
    for doc in docs:
        file_name = os.path.basename(doc.metadata.get("source", "unknown"))
        page_num = doc.metadata.get("page", "N/A")
        source_str = f"{file_name} (Page {page_num})"
        if source_str not in sources:
            sources.append(source_str)
            
    return {"answer": answer, "sources": sources}

# 4. Interactive Chat Loop
chat_history = []
print("\n--- WELCOME TO YOUR MULTI-DOCUMENT INTELLIGENCE HUB ---")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    print("AI is thinking...")
    response_data = get_response(user_input, chat_history)
    
    answer = response_data["answer"]
    sources = response_data["sources"]
    
    print(f"\nAI: {answer}")
    print("\nSources:")
    for s in sources:
        print(f"- {s}")
    
    # Update history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=answer))
