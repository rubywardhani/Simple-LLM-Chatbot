from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_runner import generate_response, reset_chat_history

app = FastAPI(
    title="Simple LLM Chatbot API",
    description="A simple chatbot API using rbGPT",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class StatusResponse(BaseModel):
    message: str


@app.get("/", response_model=StatusResponse)
def root():
    """Root endpoint untuk check API status"""
    return {"message": "Simple LLM Chatbot API is running!"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Chat endpoint to communicate with bots
    """
    try:
        if not req.message.strip():
            raise HTTPException(
                status_code=400, detail="Message cannot be empty")

        response = generate_response(req.message)

        if not response:
            response = "I'm sorry, I didn't understand that. Could you try rephrasing?"

        return {"response": response}

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error occurred")


@app.post("/reset", response_model=StatusResponse)
def reset():
    """
    Reset chat history to start a new conversation
    """
    try:
        reset_chat_history()
        return {"message": "Chat history has been reset successfully"}
    except Exception as e:
        print(f"Error in reset endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to reset chat history")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "chatbot-api"}
