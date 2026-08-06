import os
import httpx
from typing import Dict, Any, Optional

# FastAPI Backend Base URL
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class RAGAPIClient:
    """
    HTTP API client to interact with the FastAPI Adaptive RAG backend.
    """
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")

    def query(self, query_text: str, session_id: str = "default_session") -> Dict[str, Any]:
        """
        Sends a RAG query request to POST /rag/query endpoint.
        """
        url = f"{self.base_url}/rag/query"
        payload = {
            "query": query_text,
            "session_id": session_id
        }

        with httpx.Client(timeout=180.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    def upload_document(
        self,
        file_name: str,
        file_bytes: bytes,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Uploads a PDF or TXT file to POST /rag/documents/upload endpoint.
        """
        url = f"{self.base_url}/rag/documents/upload"
        headers = {"X-Description": description}
        files = {"file": (file_name, file_bytes)}

        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, files=files, headers=headers)
            response.raise_for_status()
            return response.json()