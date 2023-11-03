import chromadb
from chromadb.utils import embedding_functions
from decouple import config
import pandas as pd

# Access the API key
api_key = config('OPEN_API_KEY')


# set up the client connection to the docker container server
chroma_client = chromadb.HttpClient(host='localhost', port='8000')

print("db heartbeat: ")
print(chroma_client.heartbeat()) # returns a nanosecond heartbeat. Useful for making sure the client remains connected.

# setup the embedding function to use OpenAI ada
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-ada-002")

print("set up open ai embedding function...")

### THE COLLECTION ==================================================

# l2 is the default
# with cosing d = 1.0 - sum(Ai*Bi) / sqrt(sum(Ai*Ai) * sum(Bi*Bi))
collection = chroma_client.get_or_create_collection(name="lobbying_metadata", embedding_function=openai_ef, metadata={"hnsw:space": "cosine"}) 

# if chroma is passed a list of documents, it will automatically tokenize them and embed them with 
# the collection's embedding function, if documents are too large an exception will be raised

# each document must have a unique id associated with it
# metadata provides storage for additional information and allows for filtering

print("collection created...")