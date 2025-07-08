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

📝 Note on Question Generation

The question generation module is functional but still under improvement. It successfully generates questions from the uploaded syllabus content, but sometimes the exact number of questions per mark category (e.g., 2-mark, 5-mark) may vary slightly from the requested count.

This is due to the limitations of prompt control in local LLMs (like Mistral-7B) and token-length constraints. Efforts have been made to refine the prompts and model parameters, and further improvements can be made with more time and better hardware.


## 📂 Project Structure

```

rag-maths-chatbot/
├── main.py                 # FastAPI backend (API endpoints)
├── ingest.py               # Script to load and store syllabus PDFs
├── models/                 # Store downloaded mistral .gguf models here
├── chroma_store/           # Local vector DB (auto-created)
├── requirements.txt        # Python dependencies
├── README.md
└── ...

````

---

## 🛠️ Installation & Setup

### 🔹 1. Python Environment

Install dependencies (Python 3.10+ recommended):

```bash
pip install fastapi uvicorn langchain sentence-transformers chromadb ctransformers transformers pytesseract pillow
```

### 🔹 2. Install Tesseract-OCR (for image inputs)

* Windows: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
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

## 📌 Additional Notes

* This project was developed using Anaconda Prompt, but any terminal with Python 3.10+ and required packages will work.
* Local models can be slow on low-end machines (no GPU needed)
* Ensure `.gguf` files and `chroma_store/` are excluded via `.gitignore`
* You can re-run `ingest.py` anytime to update your vector DB
  

## 🔗 Multi-Repo Setup (Team Project)

This project is part of a team collaboration consisting of 3 separate repositories:

1. 🧠 FastAPI + RAG Backend (this repo)  
2. ☕ Spring Boot Backend (Java) – handles API communication  
3. 🌐 Frontend UI – provides user interface for students and teachers  

---

### ▶️ Steps to Run the Full System

1. First, run the FastAPI backend which handles the chatbot and question generation logic.

2. Next, run the Spring Boot backend, which acts as a bridge between the frontend and the FastAPI service.  
   Make sure it is configured with the correct FastAPI URL.

3. Finally, run the frontend application which allows users to interact with the system through a browser interface.

---

### ⚠️ Important Notes

- Start the FastAPI service **before** running the Spring Boot backend.
- The Spring Boot backend should connect to the FastAPI server through the configured URL.
- The frontend communicates only with the Spring Boot backend and not directly with FastAPI.

## Author

**KEERTHIKA K**
Backend Intern
2025
[GitHub Profile](https://github.com/KEERTHIKA-K25)
```
