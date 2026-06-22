from typing import Literal

# Field lets you add extra info or rules to a variable in model
from pydantic import BaseModel, Field

# Python class to represent a specific date and time
from datetime import datetime

# Student level
AcademicLevel = Literal[
    "highschooler",     # high school
    "undergrad_early",  # fresh, soph
    "undergrad_late",   # junior, senior
    "graduate",         # masters or phd
    "expert",           # postdoc, professor, senior researcher, domain expert
    "industry",         # professional, practical focus
    "independent"       # self-directed, background varies
]

# Why are they using this tool
ResearchGoalType = Literal[
    "understand_field",         # Student wants a broad overview of the research area and how the major ideas connect
    "join_lab_project",         # Student just joined a lab/project and needs to become useful on that specific topic
    "prepare_for_presentation", # Student has an upcoming advisor/lab presentation, meeting, defense, or interview and needs a focused briefing/questions
    "write_literature_review",  # Student needs to understand, compare, and organize papers for a written lit review
    "find_research_direction",  # Student wants to identify open problems, gaps, or possible project ideas
    "reproduce_paper"           # Student wants to understand one or more papers deeply enough to implement or reproduce them
]

# How familiar they are with the specific topic
TopicFamiliarity = Literal[
    "beginner",       # little to no knowledge of this topic
    "some_knowledge", # heard of it, know basic terms
    "familiar"        # have read papers or worked in this area
]

class ResearchGoal(BaseModel):
    topic: str                          # eg: deepfake audio detection
    goal_type: ResearchGoalType
    timeline_days: int | None = None    # eg: 2 means presentation in 2 days, None means no deadline

class StudentProfile(BaseModel):
    student_id: str
    name: str | None = None
    academic_level: AcademicLevel
    background_description: str | None = None
    topic_familiarity: TopicFamiliarity
    research_goal: ResearchGoal
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Tracks every decision agent made
class RoadmapDecision(BaseModel):
    decision_type: str   # eg: "gap_removed", "bridge_paper_added", "prep_phase_added"
    reasoning: str       # why the agent made this decision
    action_taken: str    # what actually happened as a result