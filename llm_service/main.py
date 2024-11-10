import dotenv
dotenv.load_dotenv()

from fastapi import FastAPI
from langserve import add_routes

from llm_service.ai import megachain

# Создание экземпляра приложения FastAPI
app = FastAPI()

add_routes(
    app,
    megachain,
    path="/chat", # Устанавливаем путь для доступа к чат-боту
)
