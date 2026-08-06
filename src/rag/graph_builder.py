from langgraph.graph import StateGraph, END, START
from src.models.state import GraphState
from src.rag.nodes import (
    retrieve,
    grade_documents,
    generate,
    rewrite_query,
    web_search,
)
from src.rag.nodes import route_query


# ─────────────────────────────────────────
# ROUTING FUNCTIONS (Conditional Edges)
# ─────────────────────────────────────────

def decide_after_routing(state: GraphState) -> str:
    """
    Called after route_query node.
    Returns the name of the next node to execute.
    """
    datasource = route_query(state)

    if datasource == "vectorstore":
        return "retrieve"
    elif datasource == "websearch":
        return "web_search"
    else:
        return "generate"


def decide_after_grading(state: GraphState) -> str:
    """
    Called after grade_documents node.
    If web_search flag is True, rewrite the query.
    Otherwise, proceed to generate the answer.
    """
    if state["web_search"]:
        print("--- DECISION: Documents irrelevant → Rewriting Query ---")
        return "rewrite_query"
    else:
        print("--- DECISION: Documents relevant → Generating Answer ---")
        return "generate"


# ─────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────

def build_graph():
    """
    Assembles and compiles the Adaptive RAG LangGraph StateGraph.

    Returns:
        CompiledGraph: A runnable compiled LangGraph application.
    """
    # 1. Initialize the State Graph with our GraphState schema
    graph = StateGraph(GraphState)

    # 2. Register all nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("web_search", web_search)

    # 3. Set the Entry Point with conditional routing
    # START → route_query decides the first real node
    graph.add_conditional_edges(
        START,
        decide_after_routing,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "generate": "generate",
        }
    )

    # 4. After retrieval → always grade documents
    graph.add_edge("retrieve", "grade_documents")

    # 5. After grading → conditionally go to rewrite or generate
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        }
    )

    # 6. After rewriting → always perform web search
    graph.add_edge("rewrite_query", "web_search")

    # 7. After web search → always generate
    graph.add_edge("web_search", "generate")

    # 8. After generation → END (conversation complete)
    graph.add_edge("generate", END)

    # 9. Compile the graph into a runnable application
    compiled_graph = graph.compile()

    return compiled_graph


# Singleton compiled graph instance
rag_graph = build_graph()