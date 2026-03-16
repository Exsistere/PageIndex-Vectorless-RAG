from langchain_text_splitters import RecursiveCharacterTextSplitter
from json import load
from chromadb.config import Settings
import chromadb    

def splitter(filepath: str) -> dict:
    with open(filepath, "r") as file:
        data = load(file)
    documents = {}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2500, chunk_overlap = 200)
    for node_id in data:
        documents[int(node_id)] = text_splitter.split_text(data[node_id])
    return documents

def embed_node_content(content_file):
    client = chromadb.PersistentClient(path="./chroma_db")
    print(client.list_collections())
    collection = client.get_or_create_collection(name= "sample_collection")
    
    documents = splitter(content_file)
    docs = []
    metas = []
    ids = []
    for node_id in documents:
        for i, chunk in enumerate(documents[node_id]):
            docs.append(chunk)
            metas.append({"node_id": node_id })
            ids.append(f"{node_id}_{i}")
    collection.add(
        documents=docs,
        metadatas=metas,
        ids = ids
    )
embed_node_content("content.json")