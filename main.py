import dotenv

dotenv.load_dotenv()

from fastapi import FastAPI
from langserve import add_routes

from ai_next import megachain

app = FastAPI()

add_routes(
    app,
    megachain,
    path="/chat",
)
