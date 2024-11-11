from llm import llm

from langchain_community.document_loaders import PyPDFLoader

pdf_filepath = '1706.03762v7.pdf'
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)

# print(len(pages))

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=embeddings_model)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Prompt
template = '''Answer the question based only on the following context:
{context}

Question: {question} as translate to korean
'''

prompt = ChatPromptTemplate.from_template(template)

# Rretriever
retriever = vectorstore.as_retriever()

# Combine Documents
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# RAG Chain 연결
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG")
# Chain 실행
x = rag_chain.invoke("문서 핵심을 한문장으로 요약.")
print(x)
