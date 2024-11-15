# Сервис LLM

Сервис позволяет взаимодействовать с чат-ботом, обеспечивая контекстуальные ответы на основе содержимого документов.

## :gear: Технологии

### Необходимое окружение

- **ChromaDB**

### Стек:

- **Python 3.12**
- **ChromaDB**
- **HuggingFace**
- **nltk**
- **FastAPI**
- **LangChain**
- **Ollama**
- **Pydantic**
- **dotenv**

## :hammer_and_wrench: Структура проекта

```
vector_store_manager/
├── .env
├── llm_service/
|   ├── main.py
│   ├── ai.py
│   ├── base_ai.py
│   ├── history.py
│   └── vector_store.py
├── requirements.txt
└── README.md
```

### Описание файлов

#### `main.py`

Основной файл приложения, инициализирующий веб-сервер с использованием FastAPI и добавляющий маршруты для взаимодействия с чат-ботом.

**Ключевые функции:**
- Загрузка переменных окружения с помощью `dotenv`.
- Создание экземпляра FastAPI приложения.
- Добавление маршрутов для чат-бота через `langserve`.

#### `llm_service/ai.py`

Файл отвечает за настройку цепочек LangChain, включая создание основных цепочек обработки запросов, интеграцию с моделями и ретриверами.

**Ключевые компоненты:**
- Форматирование документов.
- Определение шаблонов запросов для чат-бота.
- Настройка цепочек обработки (`megachain`, `chat_chain`, `rag_chain_from_docs` и другие).
- Интеграция с моделями Ollama и настройка ретриверов.

#### `llm_service/base_ai.py`

Файл отвечает за инициализацию языковой модели и ретривера, используемых в цепочках LangChain.

**Ключевые функции:**
- Настройка модели чата `ChatOllama`.
- Конфигурация ретривера на основе хранилища векторов с использованием метода MMR (Maximal Marginal Relevance).

#### `llm_service/vector_store.py`

Файл отвечает за настройку векторного хранилища с использованием ChromaDB и модели эмбеддингов.

**Ключевые функции:**
- Загрузка конфигураций из переменных окружения.
- Инициализация модели эмбеддингов.
- Настройка векторного хранилища Chroma.

#### `llm_service/history.py`

Файл отвечает за создание ретривера, учитывающего историю чата, что позволяет формировать контекстуальные запросы для более точных ответов.

**Ключевые функции:**
- Определение шаблона для контекстуализации вопросов.
- Создание истории-осознанного ретривера.

## :wrench: Конфигурация

Приложение настраивается при помощи переменных среды (Environment variables).

### Пример конфигурации

Пример конфигурации находится в `example.env`. Выполните эту команду, чтобы скопировать пример:

```bash
cp example.env .env
```

После чего отредактируйте файл `.env` в вашем текстовом редакторе, указав необходимые параметры

### Описание переменных окружения

- **EMBEDDING_MODEL**:
  - **Описание**: Название модели, используемой для создания эмбеддингов (векторных представлений) текстов.
  - **Значение по умолчанию**: `"BAAI/bge-m3"`.
  - **Настройка**: Можно переопределить через переменную окружения `EMBEDDING_MODEL`.

- **CHROMA_CREDS**:
  - **Описание**: Учетные данные для подключения к базе данных Chroma.
  - **Настройка**: Устанавливается через переменную окружения `CHROMA_CREDS`.

- **CHROMA_HOST**:
  - **Описание**: Хост-адрес сервера Chroma.
  - **Настройка**: Устанавливается через переменную окружения `CHROMA_HOST`.

- **CHROMA_PORT**:
  - **Описание**: Порт, на котором работает сервер Chroma.
  - **Настройка**: Устанавливается через переменную окружения `CHROMA_PORT`.

## :rocket: Развертывание

> [!Note]
> Обязательно настройте конфигурацию приложения. [Как это сделать?](#wrench-конфигурация)

### Шаги развертывания

1. **Клонирование репозитория:**
    ```bash
    git clone git@github.com:mzhn-mzhnr/ai.git
    ```

2. **Переход в директорию проекта:**
    ```bash
    cd ai
    ```

3. **Установка зависимостей:**
    Если необходима работа на GPU, добавьте соответствующий `--extra-index-url https://download.pytorch.org/whl/`
    ```bash
    pip install -r requirements.txt
    ```

4. **Запуск сервера FastAPI с помощью Uvicorn:**
    ```bash
    uvicorn llm_service.main:app --host 0.0.0.0 --port 9001
    ```

### Доступ к сервису

После успешного запуска сервера, чат-бот будет доступен по адресу:

```
http://localhost:9001/docs
```

## :scroll: Дополнительные файлы

### `requirements.txt`

Файл, содержащий список всех зависимостей проекта. Используется для установки необходимых библиотек.

### `.env`

Файл для хранения переменных окружения. Содержит конфиденциальные данные и не должен попадать в систему контроля версий.

**Пример содержимого:**
```env
EMBEDDING_MODEL=BAAI/bge-m3
CHROMA_CREDS=your_chroma_credentials
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

### `README.md`

Файл с общей информацией о проекте, инструкциями по установке и использованию.

