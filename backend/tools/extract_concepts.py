from pydantic import BaseModel
from tools.fetch_papers import Paper
import anthropic
import json

# Literal restricts a field to a fixed set of allowed values
from typing import Literal

class PaperConcepts(BaseModel):
    paper_id: str
    title: str
    research_area: str
    subfield: str
    assumes: list[str]
    introduces: list[str]
    key_methods: list[str] = []
    datasets: list[str] = []
    core_contribution: str
    difficulty_level: Literal["beginner", "intermediate", "advanced"]

def extract_concepts_batch(papers: list[Paper]) -> list[PaperConcepts]:

    # Write a detailed prompt
    papers_text = ""
    for i, paper in enumerate(papers):
        papers_text += f"""
    Paper {i}:
    Title: {paper.title}
    Abstract: {paper.abstract}
    TLDR: {paper.tldr}
    Fields of Study: {paper.fields_of_study}
    """

    prompt = f"""
    You are analyzing multiple research papers to extract structured information from each one.

    Here are {len(papers)} papers:

    {papers_text}

    For each paper extract the following and respond with ONLY a valid JSON array, no markdown, no code blocks.
    Return one object per paper in the exact same order, using this structure:

    [
        {{
            "index": 0,
            "research_area": "broad field this paper belongs to",
            "subfield": "specific niche within that field",
            "assumes": ["concept 1", "concept 2"],
            "introduces": ["new concept 1", "new concept 2"],
            "key_methods": ["method 1", "method 2"],
            "core_contribution": "one sentence summary of what makes this paper unique",
            "difficulty_level": "beginner or intermediate or advanced"
        }},
        {{
            "index": 1,
            ...
        }}
    ]

    Rules:
    - assumes: max 5 items
    - introduces: max 5 items  
    - key_methods: max 5 items
    - difficulty_level must be exactly one of: "beginner", "intermediate", "advanced"
    - Return ONLY the JSON array, nothing else
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

    # Parse response

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

    list_paper_concepts = []

    for item in data:
        original_paper = papers[item["index"]]
        paper_concepts = PaperConcepts(
            paper_id=original_paper.paper_id,
            title=original_paper.title,
            research_area=item.get("research_area", ""),
            subfield=item.get("subfield", ""),
            assumes=item.get("assumes", []),
            introduces=item.get("introduces", []),
            key_methods=item.get("key_methods", []),
            datasets=item.get("datasets", []),
            core_contribution=item.get("core_contribution", ""),
            difficulty_level=item.get("difficulty_level", "intermediate")
        )
        list_paper_concepts.append(paper_concepts)
    
    return list_paper_concepts



def extract_concepts(paper: Paper) -> PaperConcepts:
    
    # Write a detailed prompt
    prompt = f"""
    Your goal is to analyze an academic research paper to extract structured information.

    Paper Title: {paper.title}
    Paper Abstract: {paper.abstract}
    Paper Summary (TLDR): {paper.tldr}
    Fields of Study: {paper.fields_of_study}

    Extract the following information from this paper and respond with only valid JSON, no other text, no explanation, no markdown formatting, no code blocks.

    The JSON must have exactly these fields:
    - "research_area": broad field this paper belongs to (eg: "Computer Vision", "Audio Security")
    - "subfield": specific niche within that field (eg: "Image Super-Resolution", "Deepfake Audio Detection")
    - "assumes": a list of specific concepts, methods, or prior work this paper assumes the reader already understands
    - "introduces": a list of new concepts or contributions this paper introduces to the field
    - "key_methods": a list of specific technical techniques used in this paper (empty list if none)
    - "datasets": a list of datasets used in this paper (empty list if this paper doesn't use datasets, eg: theoretical papers)
    - "core_contribution": one clear sentence summarizing what makes this paper unique
    - "difficulty_level": must be exactly one of "beginner", "intermediate", or "advanced"
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

    # Parse response into PaperConcepts object

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

    paper_concepts = PaperConcepts(
        paper_id = paper.paper_id,
        title = paper.title,
        research_area = data.get("research_area", ""),
        subfield = data.get("subfield", ""),
        assumes = data.get("assumes", []),
        introduces = data.get("introduces", []),
        key_methods = data.get("key_methods", []),
        datasets = data.get("datasets", []),
        core_contribution = data.get("core_contribution", ""),
        difficulty_level = data.get("difficulty_level", "intermediate")
    )

    return paper_concepts
