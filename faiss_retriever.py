#!/usr/bin/env python
# coding: utf-8


from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pdf_parse import DataProcess
import torch


class FaissRetriever(object):
    # 初始化文档块索引，然后插入faiss库
    def __init__(self, model_path, data):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cuda"},
        )
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        del self.embeddings
        torch.cuda.empty_cache()

    # 获取top-K分数最高的文档块
    def GetTopK(self, query, k):
        context = self.vector_store.similarity_search_with_score(query, k=k)
        return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):
        return self.vector_store


if __name__ == "__main__":
    base = "."
    # model_name = base + "/m3e-large"  # better in Chinese
    model_name = base + "/bge-large-en"  # better in English
    dp = DataProcess(pdf_path=base + "/data/7215_slides.pdf")
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)

    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)

    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)

    data = dp.data

    faissretriever = FaissRetriever(model_name, data)
    faiss_ans = faissretriever.GetTopK("Which department does Dr. Zulfiqar Ali come from?", 6)
    print(faiss_ans)