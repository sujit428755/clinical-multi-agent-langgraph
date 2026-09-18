# Clinical Decision Support Multi-Agent System

A **LangGraph** multi-agent system that routes clinical questions to the
right specialist agent(s), then synthesizes a single, cited answer --
demonstrating stateful, conditional multi-agent orchestration rather than a
fixed linear pipeline.

## Why this is the flagship project

This isn't a standalone toy demo -- it **composes with the other two
projects in this portfolio**:

- The **Drug Interaction Agent** reuses Project 2's (FDA Drug Label RAG)
  retrieval layer directly as a tool, rather than re-implementing it.
- The system carries forward the same grounding/refusal discipline used in
  Project 2: agents report what they found, and the synthesis agent
  explicitly says when nothing relevant was retrieved instead of
  hallucinating an answer.

This mirrors how real enterprise AI systems get built: not as isolated
demos, but as composable services that call into each other.

## Architecture

```
                    Orchestrator (LLM-driven routing decision)
                           |
         +-----------------+-----------------+
         |                                    |
  Drug Interaction Agent              Research Agent
  (reuses Project 2's                 (Tavily web search)
   TF-IDF RAG)
         |                                    |
         +-----------------+-----------------+
                           |
                  Synthesis Agent (Groq LLM, grounded + cited)
                           |
                     final_report
```

The orchestrator dynamically decides whether a query needs the drug
interaction agent, the research agent, or both -- state (`AgentState`, a
typed dict) flows through every node via LangGraph's `StateGraph`, which is
what makes this a genuinely *stateful* multi-agent system rather than a
scripted sequence of LLM calls.

## Why Groq + Tavily

- **Groq** (`openai/gpt-oss-120b`, override with `GROQ_MODEL`) powers the orchestrator's routing
  decision and the synthesis agent's final answer -- same provider as
  Project 2, for consistency and low-latency inference.
- **Tavily** is the standard web-search tool in the LangChain/LangGraph
  ecosystem, purpose-built for AI agents (as opposed to a generic search
  API wrapper).

**Swap path:** replacing `ChatGroq` with `AzureChatOpenAI` in
`orchestrator.py` and `synthesis_agent.py` is a small, contained change --
directly relevant if targeting a role using Azure OpenAI specifically.

## Graceful offline fallbacks (and why that matters)

Every agent node degrades gracefully instead of crashing when an API key
is missing:
- No `GROQ_API_KEY` -> orchestrator uses a keyword heuristic; synthesis
  returns raw concatenated findings instead of an LLM-written summary.
- No `TAVILY_API_KEY` -> research agent returns a clear "skipped" note
  instead of failing.

This was a deliberate engineering choice, not an accident: it means the
graph's **control flow and state management** can be fully verified
without needing live API access (see "What was tested where" below), and
the system fails safely rather than crashing in front of a user.

## What was tested where

| Component | Tested |
|---|---|
| Graph topology & conditional routing (both branches) | Fully verified, offline |
| Drug Interaction Agent (TF-IDF retrieval) | Fully verified, offline (no network needed) |
| Orchestrator routing heuristic | Fully verified, offline |
| Synthesis agent (offline fallback path) | Fully verified, offline |
| Groq-powered routing & synthesis (live LLM calls) | Verify locally with GROQ_API_KEY set |
| Tavily web research (live search calls) | Verify locally with TAVILY_API_KEY set |

This project was built in a network-restricted sandbox with access to the
Groq/Tavily *domains* but no usable API keys wired up for live SDK calls.
Rather than fake it, the graph itself and every offline-testable component
were run and verified directly; the two live-API paths are structurally
complete and follow standard SDK patterns, ready to verify with real keys.

## Setup

```powershell
py -3.14 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Required: get a free key at https://console.groq.com/keys
$env:GROQ_API_KEY="gsk_..."

# Optional: get a free key at https://app.tavily.com (system works without it)
$env:TAVILY_API_KEY="tvly_..."

# CLI test
python src\graph.py "Can I take metformin with furosemide?"

# Full UI
python -m streamlit run app.py
```

## Known limitations / honest trade-offs

- **Drug interaction data is limited to the same 6 drugs from Project 2.**
  Extending the corpus (see Project 2's README) automatically extends what
  this system can answer.
- **Routing is single-shot**, not iterative -- the orchestrator makes one
  routing decision per query rather than re-planning based on intermediate
  results. A production version could add a loop where the synthesis
  agent can request more research if the first pass is insufficient.
- **No conversation memory across turns** -- each query starts a fresh
  graph run. Adding a checkpointer (LangGraph supports this natively)
  would enable multi-turn conversations.

## Interview talking points this project supports

- The difference between a prompt-chaining pipeline and a genuine stateful
  multi-agent graph with conditional branching.
- Composing RAG infrastructure (Project 2) as a callable tool inside a
  larger agent system, rather than treating each project as an island.
- Designing for graceful degradation (offline fallbacks) as a real
  engineering discipline, not just a nice-to-have.
- A clear, already-thought-through path to Azure OpenAI and multi-turn
  memory as production upgrades.
