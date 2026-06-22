# requests library lets code send HTTP requests to servers and APIs to grab data
import requests

# pydantic is a data validation library, lets you define shape of data
from pydantic import BaseModel

# Python's built-in way of interacting with operating system (work with files, folders, paths)
import os

# Reads env file and loads those variables into environment
from dotenv import load_dotenv
load_dotenv()

# Step 1: Define what a paper even looks like
class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    year: int | None
    citation_count: int
    reference_count: int | None
    authors: list[str]
    venue: str | None
    external_ids: dict | None
    open_access_pdf: str | None
    tldr: str | None
    fields_of_study: list[str] | None


# Step 2: Code the search function
def search_papers(query: str, limit: int = 10, min_year: int = 2021, min_citations: int = 5) -> list[Paper]:
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "limit": limit,
        "year": min_year,
        "minCitationCount": min_citations,
        "fields": "paperId,title,abstract,year,citationCount,referenceCount,authors,venue,externalIds,openAccessPdf,tldr,fieldsOfStudy"
    }
    headers = {
        "x-api-key": os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    }

    # Make request
    request = requests.get(url, params = params, headers = headers)

    # Get data in json format so we can work with it
    data = request.json()

    # Loop through results and build the Paper objects
    papers = []
    for item in data.get("data", []):
        try:
            paper = Paper(
                paper_id = item.get("paperId", ""),
                title = item.get("title", ""),
                abstract = item.get("abstract", "") or "",
                year = item.get("year", None),
                citation_count = item.get("citationCount", 0),
                reference_count = item.get("referenceCount"),
                authors = [author["name"] for author in item.get("authors", [])],
                venue = item.get("venue", ""),
                external_ids = item.get("externalIds"),
                open_access_pdf = item.get("openAccessPdf", {}).get("url") if item.get("openAccessPdf") else None,
                tldr = item.get("tldr", {}).get("text") if item.get("tldr") else None, 
                fields_of_study = item.get("fieldsOfStudy", [])
            )
            papers.append(paper)
        except Exception as e:
            print("Error building paper: ", e)
    
    return papers