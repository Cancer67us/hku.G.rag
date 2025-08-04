# for chatglm, pls pip install transformers==4.33.0

# 可以试试:
# what is Visual Programming Language (VPL)?
# what is App Inventor (for Android)?

# ngrok http 8001

import os
import pickle
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
# from vllm_model import ChatLLM
from rerank_model import reRankLLM
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from pdf_parse import DataProcess
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DOCUMENTS_PATH = "./data/7506_documents.pkl"

qwen7 = "./Qwen-7B-Chat"
m3e = "./m3e-large"
bge_reranker_large = "./bge-reranker-large"

tokenizer = AutoTokenizer.from_pretrained(qwen7, trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(qwen7, trust_remote_code=True)

if os.path.exists(DOCUMENTS_PATH):
    with open(DOCUMENTS_PATH, "rb") as f:
        data = pickle.load(f)
else:
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

faissretriever = FaissRetriever(m3e, data)
bm25 = BM25(data)
# llm = ChatLLM(qwen7)
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

with open("7506_web.html") as f:
    html = f.read()

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            json_request = await websocket.receive_json()
            query = json_request["query"]
            history = json_request["history"]
            start_time = time.perf_counter()

            faiss_context = faissretriever.GetTopK(query, 15)
            bm25_context = bm25.GetBM25TopK(query, 15)

            rerank_ref = reRank(rerank, 6, query, bm25_context, faiss_context)
            rerank_inputs = get_rerank(rerank_ref, query)

            response, new_history = qwen_infer(rerank_inputs, history)
            print('\n', '========the reranked input========\n', rerank_inputs, '\n')
            end_time = time.perf_counter()
            inference_time = end_time - start_time

            await websocket.send_json({
                "response": response,
                "history": new_history,
                "status": 200,
                "inference_time": round(inference_time, 4)
            })
    except WebSocketDisconnect:
        pass

def main():
    uvicorn.run(f"{__name__}:app", host="0.0.0.0", port=8001, workers=1)

if __name__ == "__main__":
    main()
