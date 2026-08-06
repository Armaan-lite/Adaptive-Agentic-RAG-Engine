from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from src.models.state import GraphState
from src.models.route_identifier import RouteQuery
from src.models.grade import GradeDocuments
from src.llms.gemini import get_llm
from src.rag.retriever_setup import get_retriever


# ─────────────────────────────────────────
# NODE 1: Query Router
# ─────────────────────────────────────────
def route_query(state: GraphState) -> str:
    """
    Analyzes the user query and routes it to the correct pipeline.
    Uses Gemini Structured Output to guarantee a valid routing decision.

    Returns:
        str: One of 'vectorstore', 'websearch', or 'general'
    """
    print("--- NODE: ROUTE QUERY ---")
    query = state["query"]

    # Bind RouteQuery schema to force structured JSON output from LLM
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)

    system_prompt = SystemMessage(content=(
        "You are an expert at routing user queries to the correct data source. "
        "Use 'vectorstore' for queries about user-uploaded documents. "
        "Use 'websearch' for queries about recent events or real-time information. "
        "Use 'general' for everything else (greetings, general knowledge, math, coding etc.)."
    ))

    result: RouteQuery = structured_llm.invoke([system_prompt, HumanMessage(content=query)])
    print(f"--- ROUTE DECISION: {result.datasource} ---")
    return result.datasource


# ─────────────────────────────────────────
# NODE 2: Document Retriever
# ─────────────────────────────────────────
def retrieve(state: GraphState) -> GraphState:
    """
    Retrieves the top-k most semantically relevant document chunks from Qdrant.
    """
    print("--- NODE: RETRIEVE DOCUMENTS ---")
    query = state["query"]

    retriever = get_retriever(k=4)
    documents = retriever.invoke(query)

    return {"documents": documents, "query": query}


# ─────────────────────────────────────────
# NODE 3: Document Grader
# ─────────────────────────────────────────
def grade_documents(state: GraphState) -> GraphState:
    """
    Evaluates each retrieved document chunk for relevance to the query.
    Irrelevant chunks are filtered out.
    If no relevant chunks remain, sets web_search=True to trigger fallback.
    """
    print("--- NODE: GRADE DOCUMENTS ---")
    query = state["query"]
    documents = state["documents"]

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(GradeDocuments)

    system_prompt = SystemMessage(content=(
        "You are a document relevance grader. "
        "Evaluate whether the retrieved document chunk contains information "
        "relevant to the user's query. "
        "Give a binary score: 'yes' if relevant, 'no' if not relevant."
    ))

    relevant_docs = []
    for doc in documents:
        result: GradeDocuments = structured_llm.invoke([
            system_prompt,
            HumanMessage(content=f"Document:\n{doc.page_content}\n\nQuery: {query}")
        ])
        if result.binary_score == "yes":
            print(f"--- DOCUMENT GRADED: RELEVANT ---")
            relevant_docs.append(doc)
        else:
            print(f"--- DOCUMENT GRADED: IRRELEVANT ---")

    # If no documents passed the relevance check, trigger web search fallback
    web_search = len(relevant_docs) == 0
    return {"documents": relevant_docs, "query": query, "web_search": web_search}


# ─────────────────────────────────────────
# NODE 4: Query Rewriter
# ─────────────────────────────────────────
def rewrite_query(state: GraphState) -> GraphState:
    """
    Rewrites the user's original query into a better-optimized search query.
    Triggered when all retrieved documents were graded as irrelevant.
    """
    print("--- NODE: REWRITE QUERY ---")
    query = state["query"]

    llm = get_llm(temperature=0)

    rewrite_prompt = [
        SystemMessage(content=(
            "You are a query optimization expert. "
            "Rewrite the user's question into a clear, concise, and search-engine optimized query. "
            "Output only the rewritten query. Nothing else."
        )),
        HumanMessage(content=f"Original query: {query}")
    ]

    response = llm.invoke(rewrite_prompt)
    rewritten_query = response.content.strip()
    print(f"--- REWRITTEN QUERY: {rewritten_query} ---")

    return {"query": rewritten_query}


# ─────────────────────────────────────────
# NODE 5: Web Search
# ─────────────────────────────────────────
def web_search(state: GraphState) -> GraphState:
    """
    Performs a real-time web search using Tavily API.
    Triggered as a fallback when Vector DB retrieval fails relevance check.
    """
    print("--- NODE: WEB SEARCH ---")
    query = state["query"]

    tavily_tool = TavilySearchResults(max_results=3)
    search_results = tavily_tool.invoke(query)

    # Convert search results into LangChain Document-like strings for consistency
    from langchain_core.documents import Document
    web_docs = [
        Document(page_content=result["content"], metadata={"source": result["url"]})
        for result in search_results
    ]

    return {"documents": web_docs, "query": query}


# ─────────────────────────────────────────
# NODE 6: Answer Generator (Grounded in Documents/Context)
# ─────────────────────────────────────────
def generate(state: GraphState) -> GraphState:
    """
    Synthesizes a final grounded answer using retrieved documents as context.
    """
    print("--- NODE: GENERATE ANSWER ---")
    query = state["query"]
    documents = state["documents"]

    # Build context string from all relevant documents
    context = "\n\n---\n\n".join([doc.page_content for doc in documents])

    llm = get_llm(temperature=0)

    generate_prompt = [
        SystemMessage(content=(
            "You are a helpful AI assistant. "
            "Answer the user's question based strictly on the provided context. "
            "If the context does not contain the answer, say so honestly. "
            "Be concise, accurate, and helpful."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]

    response = llm.invoke(generate_prompt)
    answer_text = response.content if isinstance(response.content, str) else "".join([c.get("text", "") for c in response.content if isinstance(c, dict)])
    print("--- GENERATION COMPLETE ---")

    return {"generation": answer_text}


# ─────────────────────────────────────────
# NODE 7: General LLM (Direct Answering for General/Conversational Queries)
# ─────────────────────────────────────────
def general_llm(state: GraphState) -> GraphState:
    """
    Answers general knowledge, conversational, math, or coding queries directly.
    Does NOT restrict the answer to any document context.
    """
    print("--- NODE: GENERAL LLM ---")
    query = state["query"]

    llm = get_llm(temperature=0)

    prompt = [
        SystemMessage(content=(
            "You are a helpful, accurate AI assistant. "
            "Answer the user's question directly, clearly, and concisely."
        )),
        HumanMessage(content=query)
    ]

    response = llm.invoke(prompt)
    answer_text = response.content if isinstance(response.content, str) else "".join([c.get("text", "") for c in response.content if isinstance(c, dict)])
    print("--- GENERAL LLM GENERATION COMPLETE ---")

    return {"generation": answer_text}