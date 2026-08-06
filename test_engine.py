"""
Quick Test Suite for Adaptive Agentic RAG Engine.
Validates:
1. Graph compiling and structure.
2. Max Retries & Self-Correction Logic.
"""

import sys
import unittest
from src.models.state import GraphState
from src.rag.graph_builder import decide_after_generation, rag_graph

class TestRAGEngine(unittest.TestCase):

    def test_graph_compilation(self):
        """Verify the LangGraph application compiles cleanly."""
        self.assertIsNotNone(rag_graph)
        print("[PASS] LangGraph compiled successfully.")

    def test_max_retries_safeguard(self):
        """Verify that max retries safeguard breaks out at 3 retries."""
        mock_state: GraphState = {
            "query": "Test query",
            "generation": "Test generation",
            "web_search": False,
            "documents": [],
            "session_id": None,
            "retry_count": 3
        }

        # Should immediately return 'useful' (to break out to END) when retry_count >= 3
        decision = decide_after_generation(mock_state)
        self.assertEqual(decision, "useful")
        print("[PASS] Max retry safeguard successfully triggered at limit 3.")


if __name__ == "__main__":
    unittest.main()
