"""
Quantitative RAG Evaluation Script using Datasets & pandas.
Generates evaluation dataset and saves results to CSV.
"""

import pandas as pd
from datasets import Dataset
from src.rag.graph_builder import rag_graph

def run_evaluation():
    print("--- STARTING RAG EVALUATION BENCHMARK ---")
    
    test_questions = [
        "What is Adaptive RAG?",
        "How does vector search fallback work?",
    ]
    
    questions = []
    answers = []
    contexts = []
    
    for q in test_questions:
        print(f"Evaluating Question: {q}")
        res = rag_graph.invoke({"query": q})
        
        questions.append(q)
        answers.append(res.get("generation", ""))
        raw_docs = res.get("documents", [])
        
        # Handle string or Document formats
        doc_contents = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in raw_docs]
        contexts.append(doc_contents)
        
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
    
    dataset = Dataset.from_dict(data)
    df = pd.DataFrame(data)
    
    print("\n=== EVALUATION DATASET GENERATED ===")
    print(df[['question', 'answer']])
    
    df.to_csv("rag_evaluation_results.csv", index=False)
    print("\n[SUCCESS] Saved evaluation results to rag_evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()
