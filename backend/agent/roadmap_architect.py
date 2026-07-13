# Look for modules starting from backend folder not just current folder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tools.fetch_papers import Paper, search_papers
from tools.extract_concepts import PaperConcepts, extract_concepts_batch
from tools.build_dependency_graph import build_dependency_graph, get_reading_order, find_gaps, categorize_gaps
from models.student_profile import StudentProfile, RoadmapDecision


import anthropic
import json

# LangGraph is way to build AI workflow as graph
# StateGraph is object to build that workflow
# END is constant that tells LangGraph to stop the graph
from langgraph.graph import StateGraph, END

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
    paper_concepts = extract_concepts_batch(papers)
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
    categorized_gaps = state["categorized_gaps"]
    
    # Create full text of gaps
    gaps_text = ""
    for i, gap in enumerate(categorized_gaps):
        gaps_text += f"{i}. {gap['missing_concept']} (category: {gap['category']}, assumed by: {gap['assumed_by_paper']})\n"

    # Write a detailed prompt
    prompt = f"""
    You are helping personalize a research reading roadmap for a student.

    Student Profile:
    - Academic Level: {state["student_profile"].academic_level}
    - Topic Familiarity: {state["student_profile"].topic_familiarity}
    - Background: {state["student_profile"].background_description}
    - Goal: {state["student_profile"].research_goal.goal_type}
    - Timeline: {state["student_profile"].research_goal.timeline_days} days

    Here is a list of knowledge gaps detected in the student's reading list.
    Each gap is a concept that papers assume the student knows, but no paper in the list actually teaches.

    Gaps:
    {gaps_text}

    For each gap, decide:
    1. Does this student likely already know this concept based on their background? 
    2. If they don't know it, is it worth addressing given their goal and timeline?

    Respond with ONLY valid JSON, no markdown, no code blocks.
    Return a list of objects in this exact format:

    [
        {{
            "index": 0,
            "already_known": true,
            "actionable": false,
            "reasoning": "Student mentioned knowing basic ML which covers this"
        }},
        {{
            "index": 1,
            "already_known": false,
            "actionable": true,
            "reasoning": "Student has no audio background and this is assumed by 4 papers"
        }}
    ]

    - "already_known": true if student likely knows this from their background
    - "actionable": true if this gap should be addressed (not already known and worth addressing given timeline/goal)
    - "reasoning": one sentence explaining your decision
    """

    # Call Anthropic API with prompt
    
    # client knows how to talk to Claude 
    client = anthropic.Anthropic()

    # client sends message and gets response
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # response_text is a string
    response_text = response.content[0].text

    # Strip markdown code block markers if Claude added them
    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    # json.loads converts into a dictionary
    data = json.loads(response_text)

    # build actionable gaps, decision trace, iteration count
    actionable_gaps = []
    decision_trace = []

    for item in data:
        original_gap = categorized_gaps[item["index"]]

        if item["actionable"] == True:
            actionable_gaps.append(original_gap)
        
        decision_trace.append(RoadmapDecision(
            decision_type="gap_analyzed",
            reasoning=item["reasoning"],
            action_taken="marked as actionable" if item["actionable"] == True else "ignored, student already knows this"
        ))

    iteration_count = state["iteration_count"] + 1

    return {"actionable_gaps": actionable_gaps, "decision_trace": decision_trace, "iteration_count": iteration_count}

def decide_next_step(state: RoadmapState) -> str:
    if len(state["actionable_gaps"]) > 0 and state["iteration_count"] <= 1:
        return "fill_gaps"
    return "build_roadmap"

def decide_after_fill(state: RoadmapState) -> str:
    return "build_roadmap"

def fill_gaps_node(state: RoadmapState) -> dict:
    actionable_gaps = state["actionable_gaps"]
    papers = list(state["papers"])
    bridge_papers_added = list(state["bridge_papers_added"])
    decision_trace = list(state["decision_trace"])
    topic = state["student_profile"].research_goal.topic

    # track existing paper ids to avoid duplicates
    existing_ids = {p.paper_id for p in papers}

    for gap in actionable_gaps:
        if gap["category"] == "foundational":
            decision_trace.append(RoadmapDecision(
                decision_type="foundational_gap_flagged",
                reasoning=f"{gap['missing_concept']} is foundational background knowledge not covered in your reading list",
                action_taken=f"'{gap['missing_concept']}' flagged for self-study before reading papers that assume this concept"
            ))
        elif gap["category"] == "specialized":
            query = f"{gap['missing_concept']} {topic}"
            bridge_papers = search_papers(query, limit=1)
            
            if len(bridge_papers) > 0:
                bridge_paper = bridge_papers[0]

                if bridge_paper.paper_id not in existing_ids:
                    papers.append(bridge_paper)
                    bridge_papers_added.append(bridge_paper)
                    existing_ids.add(bridge_paper.paper_id)
                    decision_trace.append(RoadmapDecision(
                        decision_type="bridge_paper_added",
                        reasoning=f"{gap['missing_concept']} was assumed by papers but never introduced",
                        action_taken=f"Added '{bridge_paper.title}' to cover '{gap['missing_concept']}'"
                    ))
                else:
                    decision_trace.append(RoadmapDecision(
                        decision_type="bridge_paper_skipped",
                        reasoning=f"'{gap['missing_concept']}' needs a bridge paper but best candidate already in list",
                        action_taken=f"Skipped duplicate: '{bridge_paper.title}'"
                    ))

            else:
                decision_trace.append(RoadmapDecision(
                    decision_type="bridge_paper_not_found",
                    reasoning=f"No suitable paper found for {gap['missing_concept']}",
                    action_taken=f"'{gap['missing_concept']}' remains unfilled - student should research independently"                ))

    return {
        "papers": papers,
        "bridge_papers_added": bridge_papers_added,
        "decision_trace": decision_trace
    }

def build_roadmap_node(state: RoadmapState) -> dict:
    student_profile = state["student_profile"]
    papers = state["papers"]
    reading_order = state["reading_order"]
    decision_trace = state["decision_trace"]
    bridge_papers_added = state["bridge_papers_added"]

    paper_lookup = {p.paper_id: p for p in papers}

    # Build the structured roadmap in a big string

    roadmap = f"""
    RESEARCH ROADMAP: {student_profile.research_goal.topic.upper()}
    Goal: {student_profile.research_goal.goal_type}
    Timeline: {student_profile.research_goal.timeline_days} days

    YOUR CUSTOMIZED READING ORDER:
    """

    for i, paper_id in enumerate(reading_order, 1):
        if paper_id in paper_lookup:
            paper = paper_lookup[paper_id]
            roadmap += f"{i}. {paper.title} ({paper.year})\n"
    
    if bridge_papers_added:
        roadmap += "\nBRIDGE PAPERS ADDED BY AGENT:\n"
        for paper in bridge_papers_added:
            roadmap += f"- {paper.title}\n"
    
    seen_foundational = set()
    foundational_flags = []
    for d in decision_trace:
        if d.decision_type == "foundational_gap_flagged":
            concept = d.action_taken
            if concept not in seen_foundational:
                seen_foundational.add(concept)
                foundational_flags.append(d)
                
    if foundational_flags:
        roadmap += "\nFOUNDATIONAL CONCEPTS TO REVIEW FIRST:\n"
        for d in foundational_flags:
            roadmap += f"- {d.reasoning}\n"
    
    roadmap += "\nAGENT DECISION TRACE:\n"
    for d in decision_trace:
        roadmap += f"- [{d.decision_type}] {d.action_taken}\n"
    
    return {"roadmap": roadmap}

def build_agent():
    
    # create empty LangGraph graph
    workflow = StateGraph(RoadmapState)

    # add all the nodes
    workflow.add_node("fetch_papers", fetch_papers_node)
    workflow.add_node("extract_concepts", extract_concepts_node)
    workflow.add_node("build_graph", build_graph_node)
    workflow.add_node("detect_gaps", detect_gaps_node)
    workflow.add_node("audit_roadmap", audit_roadmap_node)
    workflow.add_node("fill_gaps", fill_gaps_node)
    workflow.add_node("build_roadmap", build_roadmap_node)

    # set the starting node
    workflow.set_entry_point("fetch_papers")

    # add the fixed edges
    workflow.add_edge("fetch_papers", "extract_concepts")
    workflow.add_edge("extract_concepts", "build_graph")
    workflow.add_edge("build_graph", "detect_gaps")
    workflow.add_edge("detect_gaps", "audit_roadmap")
    
    # add the conditional edges
    workflow.add_conditional_edges(
        "audit_roadmap",
        decide_next_step,
        {
            "fill_gaps": "fill_gaps",
            "build_roadmap": "build_roadmap"
        }
    )
    
    workflow.add_conditional_edges(
        "fill_gaps",
        decide_after_fill,
        {
            "audit_roadmap": "audit_roadmap",
            "build_roadmap": "build_roadmap"
        }
    )
    
    workflow.add_edge("build_roadmap", END)
    
    # compiles assembled graph into runnable object
    return workflow.compile()

def run_agent(student_profile: StudentProfile) -> dict:
    agent = build_agent()
    
    initial_state = {
        "student_profile": student_profile,
        "papers": [],
        "concepts": [],
        "graph": None,
        "reading_order": [],
        "gaps": [],
        "categorized_gaps": [],
        "actionable_gaps": [],
        "bridge_papers_added": [],
        "decision_trace": [],
        "iteration_count": 0,
        "roadmap": None
    }
    
    # runs the agent starting from initial state to END
    result = agent.invoke(initial_state)
    
    return {
        "roadmap": result["roadmap"],
        "decision_trace": result["decision_trace"],
        "reading_order": result["reading_order"],
        "bridge_papers_added": result["bridge_papers_added"]
    }