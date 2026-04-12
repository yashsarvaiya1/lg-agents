from langchain_ollama import ChatOllama

chat_llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)
