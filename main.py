from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from pageindex import PageIndexClient
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Any
from loguru import logger
from treerag import TreeRAG
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


# ── Initialize TreeRAG
rag = TreeRAG(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",          # Use gpt-4o for best results, gpt-4o-mini for speed/cost
    max_pages_per_node=10,   # Max pages per tree node before splitting
    add_summaries=True,      # Generate summaries for each node
    verbose=True,            # Show progress logs
)


@app.post("/submit")
async def PDFsubmit(file: UploadFile = File(None)):
    
    pdf_path =None
    if file:
        logger.info("File imported")
        pdf_path = os.path.join("uploads", file.filename)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        tree = rag.index(
            pdf_path,
            index_path="sample_treeindex.json",   # Reuse on next run
            force_rebuild=False,
        )
        # View the tree outline 
        print("=" * 60)
        print("📋 Document Tree Outline:")
        print("=" * 60)
        print(rag.get_outline())

        
        return JSONResponse(
                                {
                                    "tree" : tree.to_dict()
                                }
                            )

@app.post("/chat")
async def chat(request: ChatRequest):
    print(query)
    query= request.query
    result = rag.query(query)
    return JSONResponse(content = result)