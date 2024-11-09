import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CHROMA_CREDS = os.getenv("CHROMA_CREDS")
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")