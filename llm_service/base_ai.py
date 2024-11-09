from llm_service.vector_store import vector_store
from llm_service.config import USE_OLLAMA, YANDEX_GPT_DIR

retriever = vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs={
        "k": 8, 
        "fetch_k": 60
    }
)

if USE_OLLAMA:
    from langchain_ollama import ChatOllama
    
    CHAT_MODEL = "krith/qwen2.5-14b-instruct:IQ4_XS"
    chat = ChatOllama(                                                                                                                                                         
        model=CHAT_MODEL,                                                                                                                                                      
        temperature=0.8,                                                                                                                                                      
        num_predict=256,                                                                                                                                                       
    )              
else:
    from langchain_community.chat_models import ChatYandexGPT
    chat = ChatYandexGPT(
        model_uri=f"gpt://{YANDEX_GPT_DIR}/yandexgpt/rc",
        temperature=0.8
    )

