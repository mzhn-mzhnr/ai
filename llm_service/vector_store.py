import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings
import llm_service.config as config

embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
)

client = chromadb.HttpClient(
    host=config.CHROMA_HOST,
    port=config.CHROMA_PORT,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
        chroma_client_auth_credentials=config.CHROMA_CREDS
    )
)

vector_store = Chroma(
    client=client,
    embedding_function=embeddings,    
)