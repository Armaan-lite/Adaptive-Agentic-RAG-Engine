from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from src.models.state import GraphState
from src.models.verification_result import GradeHallucination, GradeAnswer
from src.llms.gemini import get_llm
from src.rag.nodes import (
    retrieve,
    grade_documents,
    generate,
    rewrite_query,
    web_search,
    general_llm,
    route_query,
)


# ─────────────────────────────────────────
# ROUTING FUNCTIONS (Conditional Edges)
# ─────────────────────────────────────────

def decide_after_routing(state: GraphState) -> str:
    """
    Called after route_query.
    Returns the name of the next node to execute.
    """
    datasource = route_query(state)

    if datasource == "vectorstore":
        return "retrieve"
    elif datasource == "websearch":
        return "web_search"
    else:
        return "general_llm"


def decide_after_grading(state: GraphState) -> str:
    """
    Called after grade_documents node.
    If web_search flag is True, rewrite the query.
    Otherwise, proceed to generate the answer.
    """
    if state["web_search"]:
        print("--- DECISION: Documents irrelevant -> Rewriting Query ---")
        return "rewrite_query"
    else:
        print("--- DECISION: Documents relevant -> Generating Answer ---")
        return "generate"


def decide_after_generation(state: GraphState) -> str:
    """
    Evaluates the generated answer for hallucinations and question relevance (Self-RAG / CRAG Audit).

    Returns:
        'useful': Grounded & answers question -> END
        'not_grounded': Hallucinated -> Regenerate answer
        'not_useful': Didn't answer question -> Rewrite query & Web Search
    """
    print("--- AUDIT: EVALUATING GENERATION QUALITY ---")
    query = state["query"]
    documents = state["documents"]
    generation = state["generation"]

    llm = get_llm(temperature=0)

    # 1. Hallucination Check (Is generation grounded in facts?)
    structured_hallucination_llm = llm.with_structured_output(GradeHallucination)
    hallucination_prompt = SystemMessage(content=(
        "You are an impartial auditor evaluating hallucination. "
        "Grade whether the LLM generation is strictly grounded in and supported by the provided facts. "
        "Give binary score 'yes' if grounded, 'no' if hallucinated."
    ))
    context = "\n\n".join([doc.page_content for doc in documents])
    h_result: GradeHallucination = structured_hallucination_llm.invoke([
        hallucination_prompt,
        HumanMessage(content=f"Facts:\n{context}\n\nGeneration:\n{generation}")
    ])

    if h_result.binary_score == "yes":
        print("--- AUDIT: Generation is Grounded (No Hallucination) ---")

        # 2. Question Relevance Check (Does generation actually answer the query?)
        structured_answer_llm = llm.with_structured_output(GradeAnswer)
        answer_prompt = SystemMessage(content=(
            "You are an evaluator checking if an answer resolves a question. "
            "Grade whether the generation addresses and answers the user's question. "
            "Give binary score 'yes' if it answers the question, 'no' if not."
        ))
        a_result: GradeAnswer = structured_answer_llm.invoke([
            answer_prompt,
            HumanMessage(content=f"Question:\n{query}\n\nGeneration:\n{generation}")
        ])

        if a_result.binary_score == "yes":
            print("--- AUDIT: Answer is Useful and Resolves Query ---")
            return "useful"
        else:
            print("--- AUDIT: Answer did NOT resolve query -> Rewriting Query ---")
            return "not_useful"
    else:
        print("--- AUDIT: Generation Hallucinated -> Regenerating ---")
        return "not_grounded"


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
    graph.add_node("general_llm", general_llm)

    # 3. Entry Point Routing
    graph.add_conditional_edges(
        START,
        decide_after_routing,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "general_llm": "general_llm",
        }
    )

    # 4. Retrieval -> Grade Documents
    graph.add_edge("retrieve", "grade_documents")

    # 5. Grade Documents -> Rewrite Query OR Generate
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        }
    )

    # 6. Rewrite Query -> Web Search -> Generate
    graph.add_edge("rewrite_query", "web_search")
    graph.add_edge("web_search", "generate")

    # 7. Generate -> Self-RAG Audit (Hallucination & Question Addressal Check)
    graph.add_conditional_edges(
        "generate",
        decide_after_generation,
        {
            "useful": END,
            "not_grounded": "generate",
            "not_useful": "rewrite_query",
        }
    )

    # 8. General LLM -> END
    graph.add_edge("general_llm", END)

    # 9. Compile the graph into a runnable application
    compiled_graph = graph.compile()

    return compiled_graph


# Singleton compiled graph instance
rag_graph = build_graph()