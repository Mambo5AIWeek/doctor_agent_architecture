#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Backend Endpoint - Medical Diagnosis System
This module provides a REST API endpoint for the medical diagnosis system.
"""

import os
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
from datetime import datetime

# Import the orchestrator
import importlib.util
import sys

# Import orchestrator
spec = importlib.util.spec_from_file_location("orquestation", "orquestation.py")
orquestation_module = importlib.util.module_from_spec(spec)
sys.modules["orquestation"] = orquestation_module
spec.loader.exec_module(orquestation_module)
orchestrator = orquestation_module.orchestrator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Diagnosis API",
    description="AI-powered medical diagnosis assistance system. NOT a substitute for professional medical advice.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Session management
active_sessions: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    """Request model for chat messages."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's message describing symptoms or health concerns")
    session_id: Optional[str] = Field(None, description="Session ID for continuing conversation")

class ChatResponse(BaseModel):
    """Response model for chat messages."""
    response: str = Field(..., description="AI assistant's response")
    session_id: str = Field(..., description="Session ID for this conversation")
    conversation_complete: bool = Field(..., description="Whether the diagnosis process is complete")
    timestamp: str = Field(..., description="Response timestamp")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str
    timestamp: str

def check_environment() -> bool:
    """Check if required environment variables are set."""
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_CHAT_DEPLOYMENT", 
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    return True

def cleanup_session(session_id: str):
    """Clean up session data."""
    try:
        if session_id in active_sessions:
            orchestrator.end_session(session_id)
            del active_sessions[session_id]
            logger.info(f"Session {session_id[:8]}... cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {e}")

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    if not check_environment():
        logger.error("Application startup failed due to missing environment variables")
        raise RuntimeError("Missing required environment variables")
    
    logger.info("Medical Diagnosis API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    # Clean up all active sessions
    for session_id in list(active_sessions.keys()):
        cleanup_session(session_id)
    
    logger.info("Medical Diagnosis API shutdown complete")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Medical Diagnosis API is running",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
    background_tasks: BackgroundTasks
) -> ChatResponse:
    """
    Main chat endpoint for medical diagnosis conversations.
    
    This endpoint handles all user interactions with the medical diagnosis system.
    It processes user messages through the complete agent architecture and returns
    only the final chatbot response to the user.
    
    **Important Medical Disclaimer:**
    This system provides general health information only and is NOT a substitute 
    for professional medical advice. Always consult qualified healthcare 
    professionals for proper diagnosis and treatment.
    """
    
    try:
        # Get or create session
        session_id = request.session_id
        
        if not session_id or session_id not in active_sessions:
            # Start new session
            session_id = orchestrator.start_new_session()
            active_sessions[session_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "message_count": 0
            }
            logger.info(f"Started new session: {session_id[:8]}...")
        else:
            # Update existing session
            active_sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()
        
        # Update message count
        active_sessions[session_id]["message_count"] += 1
        
        logger.info(f"Processing message for session {session_id[:8]}... (message #{active_sessions[session_id]['message_count']})")
        
        # Process the user message through the orchestrator
        result = orchestrator.process_user_message(request.message, session_id)
        
        # Extract the response
        response_text = result.get("response", "I apologize, but I couldn't process your message. Please try again.")
        conversation_complete = result.get("conversation_complete", False)
        error = result.get("error")
        
        # Log any errors for monitoring
        if error:
            logger.error(f"Error in session {session_id[:8]}...: {error}")
        
        # Schedule session cleanup if conversation is complete
        if conversation_complete:
            background_tasks.add_task(cleanup_session, session_id)
            logger.info(f"Session {session_id[:8]}... marked for cleanup")
        
        # Create response
        response = ChatResponse(
            response=response_text,
            session_id=session_id,
            conversation_complete=conversation_complete,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Response sent for session {session_id[:8]}... (complete: {conversation_complete})")
        
        return response
        
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        
        # Return a safe error response
        error_response = """I apologize, but I'm experiencing technical difficulties at the moment. 
        
**IMPORTANT:** If you're experiencing medical symptoms, please:
- Contact your healthcare provider directly
- Call emergency services if this is urgent
- Visit an urgent care center or emergency room if needed

This AI system should never be your only source of medical guidance."""
        
        # Generate a new session ID if needed
        if not request.session_id:
            session_id = str(uuid.uuid4())
        else:
            session_id = request.session_id
        
        return ChatResponse(
            response=error_response,
            session_id=session_id,
            conversation_complete=True,
            timestamp=datetime.utcnow().isoformat()
        )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Medical Diagnosis API",
        "version": "1.0.0",
        "description": "AI-powered medical diagnosis assistance system",
        "disclaimer": "This system provides general health information only and is NOT a substitute for professional medical advice.",
        "endpoints": {
            "POST /chat": "Main chat endpoint for diagnosis conversations",
            "GET /health": "Health check endpoint",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation"
        },
        "usage": {
            "example_request": {
                "message": "I have been feeling sick with fever and cough",
                "session_id": null
            },
            "note": "Send session_id from previous responses to continue the same conversation"
        }
    }

@app.get("/sessions/active")
async def get_active_sessions():
    """Get information about active sessions (for monitoring/debugging)."""
    return {
        "active_sessions": len(active_sessions),
        "sessions": {
            session_id[:8] + "...": {
                "created_at": data["created_at"],
                "last_activity": data["last_activity"],
                "message_count": data["message_count"]
            }
            for session_id, data in active_sessions.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check environment before starting
    if not check_environment():
        print("❌ Missing required environment variables. Please check your .env file.")
        sys.exit(1)
    
    print("🏥 Starting Medical Diagnosis API Server...")
    print("⚠️  MEDICAL DISCLAIMER: This system is NOT a substitute for professional medical advice!")
    print("📚 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Alternative docs at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "frontend_endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload_dirs=["./"],
        log_level="info"
    )