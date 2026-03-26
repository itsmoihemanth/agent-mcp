from langchain_core.tools import create_retriever_tool


def make_retriever_tool(vectorstore):
    """Create a retriever tool from the vector store for the agent."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    return create_retriever_tool(
        retriever,
        name="search_documents",
        description=(
            "Search ingested documents (research papers, technical docs) for relevant information. "
            "Use this when the user asks about specific papers, technical concepts from documents, "
            "or anything that might be in the knowledge base."
        ),
    )
