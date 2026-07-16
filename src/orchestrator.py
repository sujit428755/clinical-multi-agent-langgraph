"""
Orchestrator node.

Decides which agent(s) should handle the incoming query: drug interaction
lookup, general research, both, or straight to synthesis if enough
information is already present. This is the "stateful multi-agent
architecture" piece -- routing is dynamic, driven by an LLM call reading
the shared state, not a hardcoded sequence.

Requires GROQ_API_KEY. Falls back to a simple keyword heuristic if no key
is set, so the graph is still structurally testable offline.
"""
import os
import re
from state import AgentState

GROQ_AVAILABLE = bool(os.environ.get("GROQ_API_KEY"))

DRUG_KEYWORDS = re.compile(
    r"\b(interact|interaction|take .* with|combin|contraindicat|dose|dosing|"
    r"metformin|atorvastatin|warfarin|lisinopril|sertraline|amoxicillin)\b",
    re.IGNORECASE,
)


def _heuristic_route(query: str) -> str:
    """Offline fallback: simple keyword check instead of an LLM call."""
    if DRUG_KEYWORDS.search(query):
        return "both"  # drug questions usually benefit from both label data + context
    return "research"


def orchestrator(state: AgentState) -> dict:
    query = state["query"]

    if not GROQ_AVAILABLE:
        decision = _heuristic_route(query)
        return {
            "route_decision": decision,
            "messages": [f"[orchestrator] No GROQ_API_KEY -- used keyword heuristic, routed to: {decision}"],
        }

    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = f"""You are routing a clinical question to the right specialist agent(s).

Question: {query}

Available agents:
- "drug_interaction": has grounded FDA drug label data for 6 specific drugs
  (metformin, atorvastatin, warfarin, lisinopril, sertraline, amoxicillin) --
  use for dosing, interaction, or contraindication questions involving these drugs.
- "research": general web search for current medical guidance, not limited
  to the 6-drug corpus.
- "both": use when the question needs both grounded label data AND broader context.

Reply with EXACTLY ONE WORD: drug_interaction, research, or both."""

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    if decision not in ("drug_interaction", "research", "both"):
        decision = _heuristic_route(query)  # safe fallback if LLM output is malformed

    return {
        "route_decision": decision,
        "messages": [f"[orchestrator] Routed to: {decision}"],
    }


if __name__ == "__main__":
    for q in [
        "Can I take metformin with furosemide?",
        "What are the latest CDC guidelines on diabetes screening?",
    ]:
        state: AgentState = {
            "query": q, "route_decision": "", "research_findings": [],
            "drug_interaction_findings": [], "final_report": "", "messages": [],
        }
        result = orchestrator(state)
        print(f"Query: {q}\n  -> {result['messages'][0]}\n")
