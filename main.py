from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from pageindex import PageIndexClient
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Any
from loguru import logger
import os
import getprompt
import pageindex.utils as utils
import openai
import shutil
import time
import asyncio
import json

app = FastAPI()
os.makedirs("uploads", exist_ok=True)
load_dotenv(override=True)


class ChatRequest(BaseModel):
    query: str
    node_map: Dict[str, Any]
    tree: Dict[str, Any]

pi_client = PageIndexClient(api_key= os.getenv("PAGEINDEX_API_KEY"))
async def call_llm(prompt: str, model="gpt-4.1", temperature=0):
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature= temperature
    )
    return response.choices[0].message.content.strip()

@app.post("/submit")
async def PDFsubmit(file: UploadFile = File(None)):
    
    pdf_path =None
    if file:
        logger.info("File imported")
        pdf_path = os.path.join("uploads", file.filename)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        doc_id = pi_client.submit_document(file_path= pdf_path)["doc_id"]
        logger.info(f"doc_id: {doc_id}")

        if doc_id:
            start_time = time.perf_counter()
            while not pi_client.is_retrieval_ready(doc_id):
                await asyncio.sleep(0.5)
                continue
            logger.info(f"Retrieval ready for {doc_id}. Time taken: {(time.perf_counter()-start_time):.2f}s")
            tree = pi_client.get_tree(doc_id, node_summary=True)
            tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])
            node_map = utils.create_node_mapping(tree["result"])
        return JSONResponse(
                                {
                                    "doc_id": doc_id,
                                    "tree": tree_without_text,
                                    "node_map": node_map
                                }
                            )

@app.post("/chat")
async def chat(request: ChatRequest):
    node_map= request.node_map
    tree_without_text= request.tree
    query= request.query
    
    start_time = time.perf_counter()
    search_prompt = getprompt.get_search_prompt(query, tree_without_text)
    logger.info(f"NODE SEARCH PROMPT: {search_prompt}")
    
    tree_search_result_json = await call_llm(search_prompt) ##JSON string
    retrieval_latency = time.perf_counter()-start_time

    tree_search_result = json.loads(tree_search_result_json) ##deserialize JSON string into python object e.g dict = {"thinking":..., "node_list": [...]}
    node_list = json.loads(tree_search_result_json)["node_list"]
    
    logger.info(f"Reasoning Process: {tree_search_result['thinking']}")
    logger.info(f"Retrieved Nodes: {node_list}")
    
    #####################Answer Generation####################

    
    context = "\n\n".join(node_map[node_id]["text"] for node_id in node_list)
    
    answer_prompt = f"""
        Answer the question based on the context:

        Question: {query}
        Context: {context}

        Provide a clear, concise answer based only on the context provided.
        """

    LLM_response = await call_llm(prompt=answer_prompt, temperature=1)
    response_time = time.perf_counter()-start_time
    
    logger.info(f"PROMPT: {answer_prompt}")
    logger.info(f"LLM: {LLM_response}")
    logger.info(f"Retrieval Latency: {retrieval_latency:.2f}s")
    logger.info(f"Response Time: {response_time:.2f}s")
    
    

    return JSONResponse(
        {
            "Response": LLM_response
        }
    )