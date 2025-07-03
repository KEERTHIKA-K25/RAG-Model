# 🧠 RAG-Based Maths Chatbot & Question Generator

This project implements a **Retrieval-Augmented Generation (RAG)** system using local LLMs to:

1. 💬 Let students ask syllabus-based questions and get accurate responses (Chatbot)
2. 📄 Let teachers upload syllabus content and generate exam questions (by marks)

Built with **FastAPI**, **LangChain**, **ChromaDB**, and **Mistral-7B**, this system runs completely **offline** and supports both **PDF and image inputs**.

## 🚀 Features

- ✅ **Student Chatbot**: Answer academic questions from syllabus only (not global internet)
- ✅ **Teacher Module**: Upload syllabus & auto-generate 1, 2, 5, 7, 15-mark questions
- ✅ **Local models**: Runs using `CTransformers` with no external API calls
- ✅ **Spring Boot Integrated**: Backend connected to Java-based frontend
- ✅ **PDF/Image Upload Support**: OCR via `pytesseract` for images

## 📂 Project Structure

```

rag-maths-chatbot/
├── main.py                 # FastAPI backend (API endpoints)
├── ingest.py               # Script to load and store syllabus PDFs
├── models/                 # Store downloaded mistral .gguf models here
├── chroma\_store/           # Local vector DB (auto-created)
├── requirements.txt        # Python dependencies
├── README.md
└── ...

````

---

## 🛠️ Installation & Setup

### 🔹 1. Python Environment

Install dependencies (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
````

Or manually:

```bash
pip install fastapi uvicorn langchain sentence-transformers chromadb ctransformers transformers pytesseract pillow
```

### 🔹 2. Install Tesseract-OCR (for image inputs)

* Windows: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* Or install via Chocolatey:

```bash
choco install tesseract
```

---

## 📦 Download Required Models

### 🧠 Chatbot Model (Q\&A)

Download:
[TheBloke/Mistral-7B-Instruct-v0.1-GGUF (Q2\_K)](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

Place here:

```
C:/Users/<yourname>/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.1-GGUF/...
```

### ❓ Question Generator Model

Download:
[TheBloke/Mistral-7B-Instruct-v0.2-GGUF (Q4\_K\_M)](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

Place in:

```
D:/rag-maths-chatbot/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

Update the path in `main.py` if needed.

---

## ⚙️ Run the Application

### 🔹 1. Load PDFs (Optional Step)

```bash
python ingest.py
```

This builds vector embeddings from the syllabus content and stores them in ChromaDB.

### 🔹 2. Start FastAPI Server

```bash
uvicorn main:app --reload
```

API will be available at:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🔌 API Endpoints

### `POST /ask`

* Takes a student query and returns an answer from syllabus content.

Example:

```json
{
  "query": "What is trigonometry?"
}
```

---

### `POST /upload`

* Upload a **PDF** or **image** to add its content to the vector database.

---

### `POST /generate-questions`

* Upload a syllabus and choose how many 1, 2, 5, 7, or 15 mark questions to generate.

Form fields:

* `file` (PDF or image)
* `oneMarkQuestions`, `twoMarkQuestions`, etc.

---

## 🧑‍💻 Frontend Integration

The backend is connected to a Spring Boot frontend using `RestTemplate`, which handles chat requests and displays responses in a user-friendly UI.

---

## 📌 Notes

* Local models can be slow on low-end machines (no GPU needed)
* Ensure `.gguf` files and `chroma_store/` are excluded via `.gitignore`
* You can re-run `ingest.py` anytime to update your vector DB

---

## Author

**KEERTHIKA K**
Backend Intern – RAG Developer
2025

```
