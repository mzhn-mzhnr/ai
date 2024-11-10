from llm_service.vector_store import vector_store
from langchain_ollama import ChatOllama

# Создание ретривера на основе хранилища векторов с использованием метода MMR (Maximal Marginal Relevance)
retriever = vector_store.as_retriever(
    search_type="mmr", # Тип поиска: MMR для улучшения разнообразия результатов
    search_kwargs={
        "k": 5, # Количество документов для возвращения
        "fetch_k": 60  # Количество документов для выборки перед применением MMR
    }
)
 
# Инициализация объекта ChatOllama с заданными параметрами   
CHAT_MODEL = "krith/qwen2.5-14b-instruct:IQ4_XS"
chat = ChatOllama(                                                                                                                                                         
    model=CHAT_MODEL,  # Название модели для использования                                                                                                               
    temperature=0.01,  # Параметр температуры для управления случайностью ответов                                                                                                                                                   
    num_predict=1024,  # Максимальное количество токенов для предсказания
)   

