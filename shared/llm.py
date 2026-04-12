from langchain_ollama import ChatOllama

chat_llm = ChatOllama(
    model="qwen2.5vl-3b",
    temperature=0
)
