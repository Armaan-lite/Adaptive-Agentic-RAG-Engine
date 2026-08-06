from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

# Initialize FastAPI application
app = FastAPI(
    title="Adaptive Agentic RAG Engine",
    description="Production-Grade Self-Corrective RAG System with LangGraph Multi-Agent Workflows",
    version="1.0.0",
)

# Allow cross-origin requests (needed for Streamlit frontend to call FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all RAG API routes
app.include_router(router)


@app.get("/health")
async def health_check():
    """Simple health check endpoint to verify the server is running."""
    return {"status": "healthy", "service": "Adaptive Agentic RAG Engine"}