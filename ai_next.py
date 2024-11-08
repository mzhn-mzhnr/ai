from operator import itemgetter
from ai import retriever, chat, main_prompt, format_docs
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,    
)

from langchain_core.output_parsers.string import StrOutputParser

# predict_chain = retriever | (lambda x: x[0].metadata)
    
chat_chain = RunnableParallel(
    context=lambda x: format_docs(x['documents']),
    input=itemgetter("input")
) | main_prompt | chat | StrOutputParser()

megachain = RunnableParallel(
    documents=retriever,
    input=RunnablePassthrough()
) | RunnableParallel(
    response=chat_chain,
    metainfo=lambda x: x['documents'][0].metadata
)
