"""
Drug Interaction Agent.

This node reuses the retrieval layer from Project 2 (FDA Drug Label RAG
Assistant) as a tool inside the multi-agent graph. Rather than re-scraping
or re-indexing drug label data, this agent calls into the existing,
already-tested TF-IDF retriever over the 6-drug label corpus.

This is a deliberate architecture choice: it demonstrates that the RAG
system built in Project 2 isn't just a standalone demo -- it's reusable
infrastructure another system (this multi-agent flow) can call as a tool,
which is exactly how RAG components get used in real enterprise systems.
"""
from drug_retrieve import Retriever
from state import AgentState

_retriever = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def drug_interaction_agent(state: AgentState) -> dict:
    """Retrieve relevant FDA label excerpts for the query. This is a
    retrieval-only node -- no LLM call, no network needed. Generation
    happens later in the synthesis agent, so this node is fully testable
    offline."""
    retriever = get_retriever()
    hits = retriever.retrieve(state["query"], top_k=5)

    if not hits:
        finding = "No relevant drug label information found in the indexed corpus for this query."
    else:
        lines = [f"Drug interaction findings from FDA labels (Project 2's grounded RAG):"]
        for h in hits:
            lines.append(f"- [{h.drug} :: {h.section}] {h.text[:300]}")
        finding = "\n".join(lines)

    return {
        "drug_interaction_findings": [finding],
        "messages": [f"[drug_interaction_agent] Retrieved {len(hits)} label excerpts."],
    }


if __name__ == "__main__":
    # Standalone smoke test -- fully offline, no API keys needed
    test_state: AgentState = {
        "query": "Can I take metformin with furosemide?",
        "route_decision": "",
        "research_findings": [],
        "drug_interaction_findings": [],
        "final_report": "",
        "messages": [],
    }
    result = drug_interaction_agent(test_state)
    print(result["messages"][0])
    print()
    print(result["drug_interaction_findings"][0])
