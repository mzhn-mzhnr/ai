from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from llm_service.base_ai import chat, retriever

# Системный запрос для формирования самостоятельного вопроса из истории чата и последнего вопроса пользователя
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# Создание шаблона запроса для формирования самостоятельного вопроса
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt), # Системное сообщение с инструкцией
        MessagesPlaceholder("chat_history"),       # Заполнитель для истории чата
        ("human", "{input}"),                      # Сообщение пользователя с входным вопросом
    ]
)

# Создание ретривера с учетом истории чата
history_aware_retriever = create_history_aware_retriever(
    llm=chat,                       # Используемая языковая модель
    retriever=retriever,            # Ретривер для поиска релевантных документов
    prompt=contextualize_q_prompt   # Шаблон запроса для формирования самостоятельного вопроса
)