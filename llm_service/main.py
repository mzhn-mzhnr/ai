import dotenv

dotenv.load_dotenv()

from fastapi import FastAPI
from langserve import add_routes

from llm_service.ai import megachain

app = FastAPI()

add_routes(
    app,
    megachain,
    path="/chat",
)
