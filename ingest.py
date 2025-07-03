import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
pdf_folder = "data"

all_docs = []
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, filename))
        documents = loader.load()
        all_docs.extend(documents)

print(f"✅ Loaded {len(all_docs)} documents from PDFs")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

print(f"📄 Split into {len(chunks)} chunks")

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory="chroma_store")

vectordb.persist()
print("✅ ChromaDB vector store created and saved to 'chroma_store/'")
