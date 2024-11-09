from llm_service.vector_store import vector_store
from langchain_community.chat_models import ChatYandexGPT

EMBEDDING_MODEL = "BAAI/bge-m3"
CHAT_MODEL = "krith/qwen2.5-14b-instruct:IQ4_XS"


retriever = vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs={
        "k": 8, 
        "fetch_k": 60
    }
)

chat = ChatYandexGPT(
    model_uri="gpt://b1gjp5vama10h4due384/yandexgpt/rc",
    temperature=0.8
)

