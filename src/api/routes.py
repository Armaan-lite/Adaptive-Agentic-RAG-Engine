import os
import tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File, Header
from src.models.query_request import QueryRequest
from src.rag.graph_builder import rag_graph
from src.rag.document_upload import process_and_upload_document

# Create a Router — a mini FastAPI app that groups related endpoints
router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/query")
async def query_rag(request: QueryRequest):
    """
    Accepts a user query and runs it through the Adaptive RAG LangGraph pipeline.

    Returns the AI-generated answer.
    """
    try:
        # Build the initial graph state
        initial_state = {
            "query": request.query,
            "session_id": request.session_id,
            "documents": [],
            "generation": "",
            "web_search": False,
        }

        # Run the LangGraph pipeline
        result = rag_graph.invoke(initial_state)

        return {
            "result": {
                "type": "ai",
                "content": result["generation"]
            }
        }

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="Rate limit reached on LLM API key. Please wait 10-15 seconds and try again!"
            )
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    x_description: str = Header(default="", alias="X-Description")
):
    """
    Accepts a PDF or TXT file upload, chunks and indexes it into Qdrant Vector DB.

    Returns the number of indexed chunks.
    """
    # Validate file type
    allowed_extensions = [".pdf", ".txt"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Only PDF and TXT are allowed."
        )

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Process and index into Qdrant
        num_chunks = process_and_upload_document(
            file_path=tmp_path,
            description=x_description
        )

        # Clean up temporary file
        os.unlink(tmp_path)

        return {
            "status": True,
            "filename": file.filename,
            "chunks_indexed": num_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))