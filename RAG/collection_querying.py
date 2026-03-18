from chromadb.api.models import Collection
from chromadb.api.types import QueryResult
import chromadb


def load_collection(collection_name: str) -> Collection:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)
    collection.query
    return collection

def query_collection(query: str, node_id: list[int], collection: Collection) -> dict [str: str]:
    result = collection.query(
        query_texts = query,
        n_results = 1,
        where = {"node_id": node_id[0]},
        include=["metadatas", "documents"]
    )
    # print(result["metadatas"])
    # for i in result["documents"][0]:
    #     print("---------------------")
    #     print(i)
    return {"context": "\n".join(s for s in result["documents"][0])}


# collection = load_collection("sample_collection")
# query = input("Your query: ")
# result = query_collection(query, 1, collection)
# print(result)