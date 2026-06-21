# library for representing and analyzing graphs (network of connected things)
import networkx as nx

# library that turns text into vectors so computer can measure semantic similarity
from sentence_transformers import SentenceTransformer, util
from extract_concepts import PaperConcepts

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

def get_reading_order(graph: nx.DiGraph) -> list[str]:
    reading_order = list(nx.topological_sort(graph))
    return reading_order

import time

if __name__ == "__main__":
    start_time = time.time()
    
    from fetch_papers import search_papers
    from extract_concepts import extract_concepts
    
    papers = search_papers("deepfake audio detection")
    
    print("Extracting concepts for", len(papers), "papers...")
    
    concepts_list = []
    for paper in papers[:5]:
        print("Processing:", paper.title)
        concepts = extract_concepts(paper)
        concepts_list.append(concepts)
    
    print("\nBuilding dependency graph...")
    graph = build_dependency_graph(concepts_list)
    
    print("\nNodes (papers):", graph.number_of_nodes())
    print("Edges (dependencies found):", graph.number_of_edges())
    
    print("\nDependencies found:")
    for edge in graph.edges():
        from_paper = graph.nodes[edge[0]]["title"]
        to_paper = graph.nodes[edge[1]]["title"]
        print(f" - Read '{from_paper}' before '{to_paper}'")
    
    print("\nRecommended Reading Order:")
    reading_order = get_reading_order(graph)
    for i, paper_id in enumerate(reading_order, 1):
        title = graph.nodes[paper_id]["title"]
        print(f"{i}. {title}")
    
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")