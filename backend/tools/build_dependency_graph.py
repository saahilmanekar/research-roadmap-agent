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

if __name__ == "__main__":
    print(is_similar("transformer architecture", "transformer-based models"))
    print(is_similar("transformer architecture", "reinforcement learning"))
