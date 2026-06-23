from tools.fetch_papers import Paper, search_papers
from tools.extract_concepts import PaperConcepts, extract_concepts
from tools.build_dependency_graph import build_dependency_graph, get_reading_order, find_gaps, categorize_gaps
from models.student_profile import StudentProfile, RoadmapDecision

import networkx as nx

# TypedDict is a way to describe the structure of a dictionary
# Optional: Optional[X] means value can be either X or None
from typing import TypedDict, Optional

class RoadmapState(TypedDict):
    # inputs
    student_profile: StudentProfile

    # pipeline outputs
    papers: list[Paper]
    concepts: list[PaperConcepts]
    graph: Optional[nx.DiGraph]
    reading_order: list[str]
    gaps: list[dict]
    categorized_gaps: list[dict]

    # agent decisions
    actionable_gaps: list[dict]
    bridge_papers_added: list[Paper]
    decision_trace: list[RoadmapDecision]
    iteration_count: int

    # final output
    roadmap: Optional[str]

def fetch_papers_node(state: RoadmapState) -> dict:
    topic = state["student_profile"].research_goal.topic
    timeline = state["student_profile"].research_goal.timeline_days
    goal = state["student_profile"].research_goal.goal_type
    
    # agent decides how many papers based on student context
    if timeline is not None and timeline <= 2:
        limit = 5   # tight deadline so keep it focused
    elif goal == "write_literature_review":
        limit = 20  # needs comprehensive coverage
    else:
        limit = 10  # default
    
    papers = search_papers(topic, limit=limit)
    return {"papers": papers}

def extract_concepts_node(state: RoadmapState) -> dict:
    papers = state["papers"]
    paper_concepts = []
    for paper in papers:
        paper_concepts.append(extract_concepts(paper))
    return {"concepts": paper_concepts}

def build_graph_node(state: RoadmapState) -> dict:
    concepts = state["concepts"]
    graph = build_dependency_graph(concepts)
    reading_order = get_reading_order(graph)
    return {"graph": graph, "reading_order": reading_order}

def detect_gaps_node(state: RoadmapState) -> dict:
    concepts = state["concepts"]
    gaps = find_gaps(concepts)
    categorized_gaps = categorize_gaps(gaps)
    return {"gaps": gaps, "categorized_gaps": categorized_gaps}


def audit_roadmap_node(state: RoadmapState) -> dict:
    student_profile = state["student_profile"]
    categorized_gaps = state["categorized_gaps"]

