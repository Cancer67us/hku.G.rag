# for chatglm, pls pip install transformers==4.33.0
# 可以试试:
# what is Visual Programming Language (VPL)?
# what is App Inventor (for Android)?

# python -c "import requests; print(requests.post('https://c030-116-172-93-208.ngrok-free.app/query', json={'query':'what is Visual Programming Language (VPL)?','history':[]}, headers={'ngrok-skip-browser-warning':'true'}).text)"

import os
import pickle
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from rerank_model import reRankLLM
from faiss_retriever import FaissRetriever
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from pdf_parse import DataProcess
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class QueryRequest(BaseModel):
    query: str
    history: list = []

DOCUMENTS_PATH = "./data/7506_documents.pkl"

# Model paths
qwen7 = "./Qwen-7B-Chat"
m3e = "./m3e-large"
bge_reranker_large = "./bge-reranker-large"

# Load models
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(qwen7, trust_remote_code=True)
print("Loading LLM...")
llm = AutoModelForCausalLM.from_pretrained(qwen7, trust_remote_code=True)

# Load or process documents
if os.path.exists(DOCUMENTS_PATH):
    print("Loading existing documents...")
    with open(DOCUMENTS_PATH, "rb") as f:
        data = pickle.load(f)
else:
    print("Processing PDF documents...")
    dp = DataProcess(pdf_path="./data/7506_slides.pdf")
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    data = dp.data
    with open(DOCUMENTS_PATH, "wb") as f:
        pickle.dump(data, f)

# Initialize retrievers and reranker
print("Initializing retrievers...")
faissretriever = FaissRetriever(m3e, data)
bm25 = BM25(data)
rerank = reRankLLM(bge_reranker_large)

def qwen_infer(inputs, history=None):
    response, new_history = llm.chat(tokenizer, inputs, history=history)
    return response, new_history

def get_rerank(emb_ref, query):
    prompt_template = """Based on the following known information, provide a concise and professional answer to the user's question related to the University of Hong Kong's COMP7506 Smart phone apps development course.
                        If an answer cannot be obtained from this, please say "no answer", and do not allow any creative content to be added to the answer, the answer should be in English.
                        The known content is:
                        1: {emb_ref}
                        Question:
                        {question}""".format(emb_ref=emb_ref, question=query)
    return prompt_template

def reRank(rerank, top_k, query, bm25_ref, faiss_ans):
    items = [doc for doc, score in faiss_ans]
    items.extend(bm25_ref)
    rerank_ref = rerank.predict(query, items)[:top_k]
    emb_ref = ""
    max_length = 2048
    for doc in rerank_ref:
        if len(emb_ref + doc.page_content) > max_length:
            break
        emb_ref += doc.page_content
    return emb_ref

# Load HTML for root endpoint
with open("7506_web.html") as f:
    html = f.read()

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    HTTP接口，接收JSON格式的查询请求
    示例curl命令:
    curl -X POST "http://your-ngrok-url.ngrok.io/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "what is Visual Programming Language (VPL)?", "history": []}'
    """
    try:
        query = request.query
        history = request.history
        
        if not query:
            return {"error": "Query is required", "status": 400}
        
        start_time = time.perf_counter()

        # Retrieve relevant documents
        faiss_context = faissretriever.GetTopK(query, 15)
        bm25_context = bm25.GetBM25TopK(query, 15)

        # Rerank and prepare prompt
        rerank_ref = reRank(rerank, 6, query, bm25_context, faiss_context)
        rerank_inputs = get_rerank(rerank_ref, query)

        # Generate response
        response, new_history = qwen_infer(rerank_inputs, history)
        print('\n', '========the reranked input========\n', rerank_inputs, '\n')
        
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        return {
            "response": response,
            "history": new_history,
            "status": 200,
            "inference_time": round(inference_time, 4)
        }
    
    except Exception as e:
        return {"error": str(e), "status": 500}

def main():
    uvicorn.run(f"{__name__}:app", host="0.0.0.0", port=8001, workers=1)

if __name__ == "__main__":
    main()