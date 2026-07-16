"""
Clinical Decision Support Multi-Agent System -- LangGraph orchestration.

Graph structure:

    START -> orchestrator --(route_decision)--> drug_interaction_agent -\\
                          \\                                              \\
                           `-------------------> research_agent ---------> synthesis_agent -> END
                          \\                                              /
                           `----------------------(both)-----------------

The orchestrator inspects the query and decides whether to route to the
drug interaction agent, the research agent, or both, before everything
converges on the synthesis agent. This conditional branching -- not a fixed
linear chain -- is what makes this a genuine "stateful multi-agent graph"
rather than a simple prompt pipeline.

Run: python src/graph.py "Can I take metformin with furosemide?"
"""
import sys
from langgraph.graph import StateGraph, END

from state import AgentState
from orchestrator import orchestrator
from drug_interaction_agent import drug_interaction_agent
from research_agent import research_agent
from synthesis_agent import synthesis_agent


def route_after_orchestrator(state: AgentState) -> str:
    """Conditional edge: reads the orchestrator's decision and picks the
    next node(s). LangGraph conditional edges can only return one node id
    at a time from a function like this, so 'both' fans out via a small
    trick: we route to drug_interaction first, then research always runs
    after it when 'both' was chosen (see route_after_drug_interaction)."""
    decision = state["route_decision"]
    if decision == "drug_interaction":
        return "drug_interaction_agent"
    if decision == "research":
        return "research_agent"
    return "drug_interaction_agent"  # 'both' starts with drug_interaction


def route_after_drug_interaction(state: AgentState) -> str:
    """If the orchestrator wanted both agents, go do research next;
    otherwise go straight to synthesis."""
    if state["route_decision"] == "both":
        return "research_agent"
    return "synthesis_agent"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator)
    graph.add_node("drug_interaction_agent", drug_interaction_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("synthesis_agent", synthesis_agent)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"drug_interaction_agent": "drug_interaction_agent", "research_agent": "research_agent"},
    )
    graph.add_conditional_edges(
        "drug_interaction_agent",
        route_after_drug_interaction,
        {"research_agent": "research_agent", "synthesis_agent": "synthesis_agent"},
    )
    graph.add_edge("research_agent", "synthesis_agent")
    graph.add_edge("synthesis_agent", END)

    return graph.compile()


def run(query: str):
    app = build_graph()
    initial_state: AgentState = {
        "query": query,
        "route_decision": "",
        "research_findings": [],
        "drug_interaction_findings": [],
        "final_report": "",
        "messages": [],
    }
    final_state = app.invoke(initial_state)

    print("=" * 70)
    print("EXECUTION TRACE")
    print("=" * 70)
    for m in final_state["messages"]:
        print(" ", m)

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(final_state["final_report"])

    return final_state


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Can I take metformin with furosemide?"
    run(query)
