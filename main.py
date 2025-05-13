from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

from model import generate_response

app = FastAPI()

# Cho phép frontend gọi API từ domain khác (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Kiểm tra và load FAISS index và metadata
FAISS_PATH = "data_vector/data_vector.faiss"
METADATA_PATH = "data_vector/data_vector_metadata.pkl"

@app.get("/check-vectors")
async def check_vectors():
    """Kiểm tra trạng thái của các file vector"""
    status = {
        "faiss_exists": os.path.exists(FAISS_PATH),
        "metadata_exists": os.path.exists(METADATA_PATH),
        "index_loaded": False,
        "metadata_loaded": False,
        "total_vectors": 0
    }
    
    try:
        if status["faiss_exists"] and status["metadata_exists"]:
            index = faiss.read_index(FAISS_PATH)
            with open(METADATA_PATH, "rb") as f:
                metadata = pickle.load(f)
            
            status["index_loaded"] = True
            status["metadata_loaded"] = True
            status["total_vectors"] = index.ntotal
            
        return JSONResponse(content=status)
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# Load FAISS index và metadata
print("Loading FAISS index and metadata...")
try:
    index = faiss.read_index(FAISS_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    print(f"Successfully loaded {index.ntotal} vectors")
except Exception as e:
    print(f"Error loading vectors: {str(e)}")
    print("Please run embed_data.py first to create the vector files")
    index = None
    metadata = None

# Load model embedding
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search", response_class=HTMLResponse)
async def serve_search(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.post("/search")
async def search_api(data: dict):
    if index is None or metadata is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Vector files not loaded. Please run embed_data.py first"}
        )
    query = data.get("query", "")
    try:
        # Tìm kiếm thông tin liên quan
        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        k = 1  # Only get the most similar result
        distances, indices = index.search(query_embedding, k)
        idx = indices[0][0]
        item = metadata[idx]
        
        # Tạo prompt kết hợp
        context = item.get("output", "")
        prompt = f"""Dựa trên thông tin sau đây, hãy trả lời câu hỏi của người dùng một cách chính xác và đầy đủ:

Thông tin tham khảo:
{context}

Câu hỏi của người dùng:
{query}

Hãy trả lời:"""

        # Gửi prompt đến API generate
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "")

        return JSONResponse(content={
            "results": [{
                "output": answer,
                "similarity": float(distances[0][0]),
                "context": context
            }]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"  # Gắn model cố định

# generate model
@app.post("/generate")
async def chat_api(data: dict):
    prompt = data.get("message", "")
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    return JSONResponse(content={"response": answer})

# pull model
OLLAMA_PULL_URL = "http://localhost:11434/api/pull"

@app.post("/load_model")
async def load_model(request: Request):
    data = await request.json()
    model_name = data.get("model")

    try:
        response = requests.post(OLLAMA_PULL_URL, json={"name": model_name})
        response.raise_for_status()
        return JSONResponse(content={"status": "Model loading started", "model": model_name})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat")
async def chat_api(data: dict):
    prompt = data.get("message", "")
    answer = generate_response(prompt)
    return JSONResponse(content={"response": answer})

@app.post("/debug-prompt")
async def debug_prompt(data: dict):
    if index is None or metadata is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Vector files not loaded. Please run embed_data.py first"}
        )
    query = data.get("query", "")
    try:
        # Tìm kiếm thông tin liên quan
        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        k = 1
        distances, indices = index.search(query_embedding, k)
        idx = indices[0][0]
        item = metadata[idx]
        
        # Tạo prompt kết hợp
        context = item.get("output", "")
        prompt = f"""Dựa trên thông tin sau đây, hãy trả lời câu hỏi của người dùng một cách chính xác và đầy đủ:

Thông tin tham khảo:
{context}

Câu hỏi của người dùng:
{query}

Hãy trả lời:"""

        return JSONResponse(content={
            "prompt": prompt,
            "debug_info": {
                "user_query": query,
                "retrieved_context": context,
                "similarity_score": float(distances[0][0])
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/debug", response_class=HTMLResponse)
async def serve_debug(request: Request):
    return templates.TemplateResponse("debug.html", {"request": request})


