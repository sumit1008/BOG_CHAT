# //by skm
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qa_chain import get_response  # Assuming this method gives response from vector store

app = FastAPI()

# Allow React (Vite) dev server to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class QueryRequest(BaseModel):
    question: str

# Endpoint to handle React chat query
@app.post("/chat")
async def chat(request: QueryRequest):
    response = get_response(request.question)
    return {"response": response}
