from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
import warnings


warnings.filterwarnings("ignore")

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma(persist_directory="chroma_store", embedding_function=embedding_model)

retriever = vectordb.as_retriever(search_kwargs={"k": 1})

llm = CTransformers(
    model="C:/Users/keerthi/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.1-GGUF/snapshots/731a9fc8f06f5f5e2db8a0cf9d256197eb6e05d1/mistral-7b-instruct-v0.1.Q2_K.gguf",
    model_type="mistral",
    config={"max_new_tokens": 128, "temperature": 0.5}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

print("🤖 Ask any question from your Maths book (type 'exit' to quit)")
while True:
    try:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
        result = qa_chain.invoke({"query": query})
        print(f"AI: {result['result']}")
    except Exception as e:
        print(f"[Error] {str(e)}")
