"""
Research Agent.

Uses Tavily Search (the standard web-search tool in the LangChain/LangGraph
ecosystem, purpose-built for AI agents) to pull current, general medical
guidance relevant to the query. This complements the Drug Interaction
Agent: Tavily covers broad/current medical context, while the drug
interaction agent covers strictly-grounded label data.

NOTE: requires network access + TAVILY_API_KEY. This sandbox has no
outbound network access to tavily.com, so this node is structurally
complete and follows the standard LangGraph tool-calling pattern, but its
live behavior should be verified locally (same as Groq in Project 2). Get
a free key at https://app.tavily.com
"""
import os
from state import AgentState

try:
    # Use the API wrapper directly rather than the TavilySearchResults Tool.
    # As of langchain-community 0.3.25+, TavilySearchResults hardcodes
    # response_format="content_and_artifact", so a plain .invoke({"query": ...})
    # call now returns only a summary *string* (the "content" half of a
    # ToolMessage) instead of the list[dict] of results it used to return --
    # the artifact half is silently dropped unless invoked via the full
    # tool-call protocol. The API wrapper's .results() has no such split and
    # still returns list[dict] directly, so we use that instead.
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


def research_agent(state: AgentState) -> dict:
    api_key = os.environ.get("TAVILY_API_KEY")

    if not api_key or not TAVILY_AVAILABLE:
        note = (
            "[research_agent] Skipped: no TAVILY_API_KEY set (or "
            "langchain_community not installed). Set TAVILY_API_KEY to "
            "enable live web research. Get a free key at https://app.tavily.com"
        )
        return {
            "research_findings": [note],
            "messages": [note],
        }

    try:
        wrapper = TavilySearchAPIWrapper(tavily_api_key=api_key)
        results = wrapper.results(
            query=f"clinical guidance: {state['query']}", max_results=4
        )
    except Exception as e:
        note = f"[research_agent] Tavily search failed: {e}"
        return {
            "research_findings": [note],
            "messages": [note],
        }

    if not isinstance(results, list):
        note = f"[research_agent] Unexpected Tavily response type: {type(results).__name__}"
        return {
            "research_findings": [note],
            "messages": [note],
        }

    lines = ["Web research findings:"]
    for r in results:
        if not isinstance(r, dict):
            continue
        title = r.get("title", "")
        content = r.get("content", "")[:300]
        url = r.get("url", "")
        lines.append(f"- {title}: {content}... (source: {url})")

    finding = "\n".join(lines)
    return {
        "research_findings": [finding],
        "messages": [f"[research_agent] Retrieved {len(results)} web results."],
    }


if __name__ == "__main__":
    test_state: AgentState = {
        "query": "latest guidance on metformin and kidney function monitoring",
        "route_decision": "",
        "research_findings": [],
        "drug_interaction_findings": [],
        "final_report": "",
        "messages": [],
    }
    result = research_agent(test_state)
    print(result["messages"][0])
    print(result["research_findings"][0])
