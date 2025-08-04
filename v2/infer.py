
import os
import pickle
import time
from src.retriever.bm25_retriever import BM25
from src.retriever.tfidf_retriever import TFIDF
from src.retriever.faiss_retriever import FaissRetriever
from src.retriever.milvus_retriever import MilvusRetriever 
from src.client.llm_local_client import request_chat
from src.client.llm_hyde_client import request_hyde
from src.reranker.bge_m3_reranker import BGEM3ReRanker 
from src.constant import bge_reranker_tuned_model_path
from src.utils import merge_docs, post_processing

bm25_retriever = BM25(docs=None, retrieve=True)
milvus_retriever = MilvusRetriever(docs=None, retrieve=True) 
bge_m3_reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path)
milvus_retriever.retrieve_topk("this is a test query", topk=3)


while True:
    query = input("INPUT—>")

    t1 = time.time()
    bm25_docs = bm25_retriever.retrieve_topk(query, topk=10)
    print("BM25 recall samples:")
    print(bm25_docs)
    print("="*100)
    t2 = time.time()

    milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
    print("BGE-M3 recall samples:")
    print(milvus_docs)
    print("="*100)
    t3 = time.time()

    merged_docs = merge_docs(bm25_docs, milvus_docs)
    print(merged_docs)
    print("="*100)

    ranked_docs = bge_m3_reranker.rank(query, merged_docs, topk=5)
    print(ranked_docs)
    print("="*100)

    context = "\n".join(["【" + str(idx+1) + "】" + doc.page_content for idx, doc in enumerate(ranked_docs)])
    res_handler = request_chat(query, context, stream=True)
    response = ""
    for r in res_handler:
        uttr = r.choices[0].delta.content
        response += uttr 
        print(uttr, end='')
    print("\n" + "="*100)

    answer = post_processing(response, ranked_docs)
    print("\nANSWER—>", answer)

