from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import warnings
import shutil
import os
import io
from PIL import Image
import pytesseract
from transformers import AutoTokenizer
import re

warnings.filterwarnings("ignore")

app = FastAPI()

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_store", embedding_function=embedding_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 1})

# Models
small_llm = CTransformers(
    model="C:/Users/keerthi/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.1-GGUF/snapshots/731a9fc8f06f5f5e2db8a0cf9d256197eb6e05d1/mistral-7b-instruct-v0.1.Q2_K.gguf",
    model_type="mistral",
    config={"max_new_tokens": 256, "temperature": 0.5}
)

large_llm = CTransformers(
    model="C:/Users/keerthi/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.1-GGUF/snapshots/731a9fc8f06f5f5e2db8a0cf9d256197eb6e05d1/mistral-7b-instruct-v0.1.Q2_K.gguf",
    model_type="mistral",
    config={"max_new_tokens": 512, "temperature": 0.7}
)

qgen_llm = CTransformers(
    model="C:/Users/keerthi/Downloads/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    config={"max_new_tokens": 1024, "temperature": 0.9, "context_length": 8192}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=small_llm,
    retriever=retriever,
    return_source_documents=False
)

class AskRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(payload: AskRequest):
    try:
        query = payload.query
        retrieved_docs = retriever.get_relevant_documents(query)
        if not retrieved_docs:
            return {"response": "Sorry, I couldn't find any relevant information in the uploaded content."}
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        prompt = (
            "Answer the question using only the information provided in the context.\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        response = large_llm(prompt)
        return {"response": response.strip()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    try:
        if filename.endswith(".pdf"):
            temp_file_path = f"temp_{os.path.basename(file.filename)}"
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            os.remove(temp_file_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
            vectordb.add_documents(chunks)
            vectordb.persist()
            return {"message": f"Successfully ingested {len(chunks)} chunks from PDF {file.filename}"}
        elif file.content_type.startswith("image/"):
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            extracted_text = pytesseract.image_to_string(image)
            if not extracted_text.strip():
                return {"error": "No text found in image."}
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            text_chunks = splitter.split_text(extracted_text)
            documents = [Document(page_content=chunk) for chunk in text_chunks]
            vectordb.add_documents(documents)
            vectordb.persist()
            return {"message": f"Text extracted and added to vector store from image. Added {len(documents)} chunks."}
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF or image files only.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-questions")
async def generate_questions(
    file: UploadFile = File(...),
    oneMarkQuestions: int = Form(0),
    twoMarkQuestions: int = Form(0),
    fiveMarkQuestions: int = Form(0),
    sevenMarkQuestions: int = Form(0),
    fifteenMarkQuestions: int = Form(0),
):
    try:
        filename = file.filename.lower()
        content_text = ""

        # Extract content from PDF or image
        if filename.endswith(".pdf"):
            temp_file_path = f"temp_qgen_{os.path.basename(file.filename)}"
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            os.remove(temp_file_path)
            content_text = "\n".join([doc.page_content for doc in documents])

        elif file.content_type.startswith("image/"):
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            content_text = pytesseract.image_to_string(image)
        else:
            raise HTTPException(status_code=400, detail="❌ Unsupported file type.")

        content_text = content_text.strip()
        if not content_text:
            raise HTTPException(status_code=400, detail="❌ No readable text found.")

        # Chunk the content
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokens = tokenizer.encode(content_text, truncation=False)
        max_tokens_per_chunk = 280
        token_chunks = [tokens[i:i+max_tokens_per_chunk] for i in range(0, len(tokens), max_tokens_per_chunk)]
        text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]

        # Function to generate questions of a specific mark
        def generate_by_mark(mark, count):
            questions = []
            pattern = re.compile(rf"^Q\d+\..*?\({mark} Mark", re.IGNORECASE)

            for chunk in text_chunks:
                if len(questions) >= count:
                    break
                prompt = (
                    f"You are an AI that generates exactly {count} exam questions of {mark} Marks "
                    f"only from the content below.\n"
                    f"Format: Q<number>. <question>? ({mark} Mark{'s' if mark > 1 else ''})\n\n"
                    f"CONTENT:\n{chunk}"
                )
                try:
                    result = qgen_llm(prompt).strip()
                    lines = [line.strip() for line in result.split('\n') if pattern.match(line)]
                    questions.extend(lines)
                    questions = list(dict.fromkeys(questions))  # Remove duplicates
                except Exception as e:
                    print(f"Chunk error (mark {mark}):", e)

            return questions[:count]

        # Call function for each type
        all_questions = []
        all_questions += generate_by_mark(1, oneMarkQuestions)
        all_questions += generate_by_mark(2, twoMarkQuestions)
        all_questions += generate_by_mark(5, fiveMarkQuestions)
        all_questions += generate_by_mark(7, sevenMarkQuestions)
        all_questions += generate_by_mark(15, fifteenMarkQuestions)

        if not all_questions:
            raise HTTPException(status_code=500, detail="❌ No valid questions generated. Try uploading a better file.")

        return {"questions": "\n\n".join(all_questions)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))