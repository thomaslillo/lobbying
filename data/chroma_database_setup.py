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
collection = chroma_client.get_or_create_collection(name="ea_metadata", embedding_function=openai_ef, metadata={"hnsw:space": "cosine"}) 

# if chroma is passed a list of documents, it will automatically tokenize them and embed them with 
# the collection's embedding function, if documents are too large an exception will be raised

# each document must have a unique id associated with it
# metadata provides storage for additional information and allows for filtering

print("collection created...")

### THE DATA PREP =====================================================

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('Data/VariableMetadata2022.csv')

print("starting data write...")

### WRITING THE DATA ==================================================

# use batches of 1000 so as not to overwhelm the OpenAI API
batch_size = 1000
num_batches = (len(df) + batch_size - 1) // batch_size

for batch_num in range(num_batches):
    print("on batch" + str(batch_num) + " of " + str(num_batches) + "...")
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, len(df))
    batch = df.iloc[start_index:end_index]

    documents = []
    metadatas = []
    ids = []

    for index, row in batch.iterrows():
        r_document = row['VARIABLEDESC']
        r_metadata = {}
        r_metadata['COUNTRY'] = row['COUNTRY']
        r_metadata['DATASETNAME'] = row['DATASETNAME']
        r_metadata['VINTAGE'] = row['VINTAGE']
        r_metadata['VARIABLENAME'] = row['VARIABLENAME']
        r_id = str(row['UniqueId'])
        documents.append(r_document)
        metadatas.append(r_metadata)
        ids.append(r_id)

    try:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )  
    except:
        print("\nError in batch: " + str(batch_num))
        print("IDs starting at " + str(start_index) + " and ending at " + str(end_index) + "... \n")

print("\n\n...done writing data!")