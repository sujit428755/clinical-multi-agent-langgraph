"""
Streamlit UI for the Clinical Decision Support Multi-Agent System.
Run: streamlit run app.py
"""
import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from graph import build_graph

st.set_page_config(page_title="Clinical Decision Support Agent", page_icon="🩺", layout="centered")

st.title("🩺 Clinical Decision Support Multi-Agent System")
st.caption(
    "A LangGraph multi-agent system: an orchestrator routes each question to a "
    "grounded FDA drug-label agent (reusing Project 2's RAG), a web research agent, "
    "or both -- then a synthesis agent produces a cited final answer."
)

with st.expander("ℹ️ About this architecture"):
    st.markdown("""
    **Agents:**
    - **Orchestrator** -- reads the question, decides which specialist agent(s) to invoke
    - **Drug Interaction Agent** -- grounded retrieval over 6 FDA drug labels (reuses Project 2)
    - **Research Agent** -- live web search via Tavily for broader medical context
    - **Synthesis Agent** -- combines findings into one cited, structured answer

    **Requires:** `GROQ_API_KEY` (orchestrator + synthesis reasoning) and optionally
    `TAVILY_API_KEY` (web research -- the system still works without it, just skips
    that agent).
    """)

groq_key_set = bool(os.environ.get("GROQ_API_KEY"))
tavily_key_set = bool(os.environ.get("TAVILY_API_KEY"))

col1, col2 = st.columns(2)
col1.metric("GROQ_API_KEY", "✅ Set" if groq_key_set else "❌ Missing")
col2.metric("TAVILY_API_KEY", "✅ Set" if tavily_key_set else "⚠️ Optional")

if not groq_key_set:
    st.warning(
        "GROQ_API_KEY isn't set -- the system will still run using offline "
        "fallbacks (keyword routing, raw findings instead of LLM synthesis) "
        "so you can see the graph structure, but for real LLM-powered "
        "routing/synthesis, set the environment variable and restart."
    )

query = st.text_input(
    "Ask a clinical question:",
    placeholder="e.g. Can I take metformin with furosemide?",
)

example_queries = [
    "Can I take metformin with furosemide?",
    "Can atorvastatin be taken with clarithromycin?",
    "Is amoxicillin safe with warfarin?",
    "What are general lifestyle tips for managing diabetes?",
]
st.caption("Try: " + " | ".join(f"`{q}`" for q in example_queries))

if st.button("🔍 Ask the Agent System", type="primary", use_container_width=True) and query:
    with st.spinner("Running multi-agent graph..."):
        app = build_graph()
        initial_state = {
            "query": query,
            "route_decision": "",
            "research_findings": [],
            "drug_interaction_findings": [],
            "final_report": "",
            "messages": [],
        }
        final_state = app.invoke(initial_state)

    st.subheader("Execution Trace")
    for m in final_state["messages"]:
        st.text(m)

    st.subheader("Final Report")
    st.markdown(final_state["final_report"])
