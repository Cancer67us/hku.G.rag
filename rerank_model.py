from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch

from bm25_retriever import BM25
from pdf_parse import DataProcess
from config import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


# 释放gpu上没有用到的显存以及显存碎片
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# 加载rerank模型
class reRankLLM(object):
    def __init__(self, model_path, max_length=512, device="cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)  # Move to GPU 1 first
        self.model.half()  # Apply FP16 after moving to device
        self.model.eval()
        self.max_length = max_length
        # Verify all parameters are on the correct device
        for param in self.model.parameters():
            assert param.device == torch.device(device), f"Parameter on wrong device: {param.device}"
        for buffer in self.model.buffers():
            assert buffer.device == torch.device(device), f"Buffer on wrong device: {buffer.device}"

    def predict(self, query, docs):
        pairs = [(query, doc.page_content) for doc in docs]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)  # Tokenized inputs to GPU 1
        with torch.no_grad():
            scores = self.model(**inputs).logits
        scores = scores.detach().cpu().clone().numpy()
        response = [
            doc
            for _, doc in sorted(
                zip(scores, docs), reverse=True, key=lambda x: x[0]
            )
        ]
        torch_gc()  # Clean up GPU memory
        return response


if __name__ == "__main__":
    bge_reranker_large = "./bge-reranker-large"
    rerank = reRankLLM(bge_reranker_large)
