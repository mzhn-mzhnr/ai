import os
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

pdf_directory = "./media_all/"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


documents = []

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

stop_words = set(
    stopwords.words('russian')    
    + stopwords.words('english')
)

import regex as re

en_stemmer = PorterStemmer()
ru_stemmer = SnowballStemmer("russian")

def preprocess_text(text):
    text = text.lower()
    
    text = re.sub(r"[^\w\s\p{Sc}%#@&*()+/\-]+", "", text, flags=re.UNICODE)
    
    words = word_tokenize(text)
    
    processed_words = []
    for word in words:
        if word not in stop_words:
            if re.match(r'[а-яё]', word): 
                lemma = ru_stemmer.stem(word)
            elif re.match(r'^[a-z]+$', word):
                lemma = en_stemmer.stem(word)
            else:
                lemma = word
            processed_words.append(lemma)

    return ' '.join(processed_words)


# Iterate through each PDF file in the directory
load_start = time.time()
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        # Load the PDF file
        loader = PyMuPDFLoader(os.path.join(pdf_directory, filename))
        pages = [page.page_content for page in loader.lazy_load()]

        # Process each page, split the text, and add metadata
        page_number = 1
        for page in pages:
            if not page.strip():
                page_number += 1
                continue
            
            preprocessed_page = preprocess_text(page)
            
            # Split text into chunks
            splits = text_splitter.split_text(page)
            
            # Add each split to the documents list with metadata
            for split in splits:
                documents.append(
                    Document(
                        page_content=split,
                        metadata={
                            "page_number": page_number,
                            "filename": filename
                        }
                    )
                )
            page_number += 1
load_end = time.time()

print(f"Time to load and process PDFs: {load_end - load_start:.2f} seconds")

embedding_start = time.time()
EMBEDDING_MODEL = "BAAI/bge-m3"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
)
embedding_end = time.time()
print(f"Time to initialize embeddings: {embedding_end - embedding_start:.2f} seconds")

chroma_start = time.time()
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma"
)
chroma_end = time.time()
print(f"Time to initialize Chroma vector store: {chroma_end - chroma_start:.2f} seconds")

print("================")
print(f"Total documents prepared: {len(documents)}")

add_docs_start = time.time()
for i in range(0, len(documents), 5000):
    vector_store.add_documents(documents=documents[i:i+5000])
    print(f"Added documents from index {i} to {i + 5000}")
add_docs_end = time.time()
print(f"Time to add all documents to vector store: {add_docs_end - add_docs_start:.2f} seconds")

print("================")
print("All operations completed.")