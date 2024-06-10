# Toronto Lobbying Search

#### Pulling lobbying activity data from Toronto Open Data (ETL) into a vector database (ChromaDB) and allowing users to search through it via a webapp (streamlit).

Semantic matching is matching search queries to keywords based upon the intent of what the searcher typed into the engine, instead of just using keywords.

Source of the data: https://open.toronto.ca/dataset/lobbyist-registry/

TO DO:
- UI
- dockerize pipeline

## Running ChromaDB

You can run ChromaDB within a docker container by pulling the official image and running it.

```
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
```

## Cost Estimation
Done using: 
https://github.com/openai/tiktoken
