# api_gateway.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents import agents_system

app = FastAPI(title="AetherNet API Gateway")

class ChatRequest(BaseModel):
    message: str
    user_id: str
    role: str = ""
    interests: str = ""

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_data = {"role": request.role, "interests": request.interests}
        response_data = agents_system.process_query(request.message, user_data)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AetherNet API Gateway"}