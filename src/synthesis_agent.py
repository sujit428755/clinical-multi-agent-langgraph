"""
Synthesis Agent.

Combines whatever the research agent and/or drug interaction agent found
into a single, structured, cited answer. Mirrors the grounding/refusal
discipline from Project 2: if neither agent found anything relevant, this
agent says so explicitly instead of inventing an answer from the LLM's
general knowledge.

Requires GROQ_API_KEY for the actual synthesis. Falls back to a raw
concatenation of findings if no key is set, so the graph is still
structurally testable offline.
"""
import os
from state import AgentState

GROQ_AVAILABLE = bool(os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a clinical information synthesis assistant. You combine \
findings from a drug-label retrieval agent and/or a web research agent into a single, \
clear, well-organized answer.

Rules:
1. Only use the findings provided below -- do not add outside medical knowledge.
2. Clearly separate what came from grounded FDA drug labels vs. general web research,
   since they carry different levels of authority.
3. If the findings conflict, state the conflict explicitly.
4. If no relevant findings were provided at all, say so clearly and recommend
   consulting a pharmacist or clinician -- do not guess.
5. Never give a specific dosing recommendation for an individual patient.
6. Keep the answer concise and structured with short sections."""


def synthesis_agent(state: AgentState) -> dict:
    drug_findings = state.get("drug_interaction_findings", [])
    research_findings = state.get("research_findings", [])

    if not drug_findings and not research_findings:
        report = (
            "No information was retrieved from either the drug label database "
            "or web research for this query. Please consult a pharmacist or "
            "clinician, or rephrase the question."
        )
        return {"final_report": report, "messages": ["[synthesis_agent] No findings to synthesize."]}

    if not GROQ_AVAILABLE:
        # Offline fallback: just concatenate raw findings, no LLM synthesis
        parts = ["=== SYNTHESIS (offline fallback -- no LLM call) ==="]
        if drug_findings:
            parts.append("\n".join(drug_findings))
        if research_findings:
            parts.append("\n".join(research_findings))
        report = "\n\n".join(parts)
        return {"final_report": report, "messages": ["[synthesis_agent] No GROQ_API_KEY -- returned raw findings."]}

    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=1000)

    context_parts = []
    if drug_findings:
        context_parts.append("GROUNDED FDA DRUG LABEL FINDINGS:\n" + "\n".join(drug_findings))
    if research_findings:
        context_parts.append("WEB RESEARCH FINDINGS:\n" + "\n".join(research_findings))
    context = "\n\n".join(context_parts)

    user_message = f"QUESTION: {state['query']}\n\n{context}\n\nSynthesize a final answer following the system rules."

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    return {
        "final_report": response.content,
        "messages": ["[synthesis_agent] Synthesized final report via Groq."],
    }


if __name__ == "__main__":
    test_state: AgentState = {
        "query": "Can I take metformin with furosemide?",
        "route_decision": "both",
        "research_findings": [],
        "drug_interaction_findings": [
            "Drug interaction findings: [METFORMIN :: DRUG INTERACTIONS] Furosemide increases metformin Cmax (+22%) and AUC (+15%)."
        ],
        "final_report": "",
        "messages": [],
    }
    result = synthesis_agent(test_state)
    print(result["messages"][0])
    print()
    print(result["final_report"])
