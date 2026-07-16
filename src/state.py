"""
Shared state definition for the Clinical Decision Support multi-agent system.

Using LangGraph's typed state pattern: every node reads from and writes to
this shared state object as the graph executes. This is what "stateful
multi-agent architecture" (from the job description) actually refers to --
as opposed to a simple prompt-chaining pipeline, agents here share and build
on a common, typed state across the whole run.
"""
from typing import TypedDict, Annotated, Literal
import operator


class AgentState(TypedDict):
    query: str                              # the original user question
    route_decision: str                     # which agent(s) the orchestrator picked
    research_findings: Annotated[list[str], operator.add]   # web research notes
    drug_interaction_findings: Annotated[list[str], operator.add]  # from Project 2's RAG
    final_report: str                       # synthesized answer
    messages: Annotated[list[str], operator.add]  # trace log of what each agent did


RouteDecision = Literal["research", "drug_interaction", "both", "synthesize"]
