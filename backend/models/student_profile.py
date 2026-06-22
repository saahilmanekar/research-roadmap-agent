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
    "postdoc",          # postdoc, deep expertise
    "industry",         # professional, practical focus
    "independent"       # self-directed, background varies
]

# Why are they using this tool
ResearchGoalType = Literal[
    "understand_field",        # Student wants a broad overview of the research area and how the major ideas connect
    "join_lab_project",        # Student just joined a lab/project and needs to become useful on that specific topic
    "prepare_for_meeting",     # Student has an upcoming advisor/lab meeting and needs a focused briefing/questions
    "write_literature_review", # Student needs to understand, compare, and organize papers for a written lit review
    "find_research_direction", # Student wants to identify open problems, gaps, or possible project ideas
    "reproduce_paper"          # Student wants to understand one or more papers deeply enough to implement or reproduce them
]