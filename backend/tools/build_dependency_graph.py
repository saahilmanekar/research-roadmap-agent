import anthropic
import json

# library for representing and analyzing graphs (network of connected things)
import networkx as nx

# library that turns text into vectors so computer can measure semantic similarity
from sentence_transformers import SentenceTransformer, util
from tools.extract_concepts import PaperConcepts

# standard, basic model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to determine if 2 phrases are similar
def is_similar(phrase1: str, phrase2: str, threshold: float = 0.6) -> bool:
    
    # Convert both phrases into embeddings
    p1_embedding = model.encode(phrase1)
    p2_embedding = model.encode(phrase2)
    
    # Calculate similarity score
    score = util.cos_sim(p1_embedding, p2_embedding)
    similarity_value = score.item()

    if score >= threshold:
        return True
    else:
        return False

def build_dependency_graph(concepts_list: list[PaperConcepts]) -> nx.DiGraph:
    
    # Create Directed Graph which has nodes (vertices) and connections (edges)
    graph = nx.DiGraph()

    introduces_phrases = []
    introduces_paper_ids = []
    assumes_phrases = []
    assumes_paper_ids = []

    # Add each concept, populate lists
    for concept in concepts_list:
        graph.add_node(concept.paper_id, title=concept.title)

        for phrase in concept.introduces:
            introduces_phrases.append(phrase)
            introduces_paper_ids.append(concept.paper_id)
        
        for phrase in concept.assumes:
            assumes_phrases.append(phrase)
            assumes_paper_ids.append(concept.paper_id)
    
    # Compares each paper against every other paper
    for concept_a in concepts_list:
        for concept_b in concepts_list:
            if concept_a == concept_b:
                continue
            for introduced_phrase in concept_a.introduces:
                for assumed_phrase in concept_b.assumes:
                    if is_similar(introduced_phrase, assumed_phrase):
                        graph.add_edge(concept_a.paper_id, concept_b.paper_id)
    
    return graph

# Create order using topological sort
def get_reading_order(graph: nx.DiGraph) -> list[str]:
    try:
        reading_order = list(nx.topological_sort(graph))
        return reading_order
    except:
        print("WARNING: Cycle detected in dependency graph, using fallback ordering")
        return list(graph.nodes())

def find_gaps(concepts_list: list[PaperConcepts]) -> list[dict]:
    
    gaps = []
    
    # gather all introduced concepts
    all_introduced = []
    for concept in concepts_list:
        for phrase in concept.introduces:
            all_introduced.append(phrase)
    
    # loop through every paper's assumes list, check if each phrase is covered somewhere in all_introduced
    for concept in concepts_list:
        for assumed_phrase in concept.assumes:

            is_covered = False
            for introduced_phrase in all_introduced:
                if is_similar(assumed_phrase, introduced_phrase):
                    is_covered = True
                    break

            if not is_covered:
                gaps.append({
                    "missing_concept": assumed_phrase,
                    "assumed_by_paper": concept.title
                })
    return gaps

def categorize_gaps(gaps: list[dict]) -> list[dict]:

    # Build a list of gap concepts to send to Claude
    concepts_list_text = ""
    
    for index, gap in enumerate(gaps):
        concepts_list_text += f"{index}. {gap['missing_concept']}\n"

    # Write a detailed prompt
    prompt = f"""
    You are classifying research concepts by how foundational versus specialized they are.

    Here is a numbered list of concepts. For each one, classify it as either:
    - "foundational": a widely-taught concept easily found in standard textbooks, intro courses, or general online tutorials (eg: "Convolutional Neural Networks", "linear algebra basics", "gradient descent")
    - "specialized": a niche, field-specific concept that would be hard to learn outside of research papers in this exact subfield (eg: a specific named dataset, benchmark, or technique unique to this research area)

    Concepts: 
    {concepts_list_text}

    Respond with ONLY valid JSON, no markdown, no code blocks, no explanation. 
    The JSON must be a list of objects, one per concept, in the exact same order as given above, with this structure:

    [
        {{"index": 0, "category": "foundational"}},
        {{"index": 1, "category": "specialized"}},
        ...
    ]

    The "index" must match the number from the list above. The "category" must be exactly "foundational" or "specialized".
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

    # modify original gaps
    for item in data:
        gap_index = item["index"]
        gaps[gap_index]["category"] = item["category"]
    
    return gaps