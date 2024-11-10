import os

# Получение модели эмбеддингов из переменной окружения или установка значения по умолчанию
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
# Получение учетных данных для Chroma из переменной окружения
CHROMA_CREDS = os.getenv("CHROMA_CREDS")
# Получение хоста для Chroma из переменной окружения
CHROMA_HOST = os.getenv("CHROMA_HOST")
# Получение порта для Chroma из переменной окружения
CHROMA_PORT = os.getenv("CHROMA_PORT")
