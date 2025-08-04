import json
import time

from vllm_model import ChatLLM
from rerank_model import reRankLLM
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from pdf_parse import DataProcess


def get_emb_bm25_merge(faiss_context, bm25_context, query):
    max_length = 2500
    emb_ref = ""
    cnt = 0
    for doc, score in faiss_context:
        cnt = cnt + 1
        if cnt > 6:
            break
        if len(emb_ref + doc.page_content) > max_length:
            break
        emb_ref = emb_ref + doc.page_content
    bm25_ref = ""
    cnt = 0
    for doc in bm25_context:
        cnt = cnt + 1
        if len(bm25_ref + doc.page_content) > max_length:
            break
        bm25_ref = bm25_ref + doc.page_content
        if cnt > 6:
            break

    prompt_template = """Based on the following known information, provide a concise and professional answer to the user's question related to the University of Hong Kong's IDAT7215 Computer programming for product development and applications course.
                        If an answer cannot be obtained from this, please say "no answer" or "no answer", and do not allow any creative content to be added to the answer, the answer should be in English.
                                The known content is:
                                1: {emb_ref}
                                2: {bm25_ref}
                                Question:
                                {question}""".format(
        emb_ref=emb_ref, bm25_ref=bm25_ref, question=query
    )
    return prompt_template


def get_rerank(emb_ref, query):

    prompt_template = """Based on the following known information, provide a concise and professional answer to the user's question related to the University of Hong Kong's IDAT7215 Computer programming for product development and applications course.
                        If an answer cannot be obtained from this, please say "no answer" or "no answer", and do not allow any creative content to be added to the answer, the answer should be in English.
                                The known content is:
                                1: {emb_ref}
                                Question:
                                {question}""".format(
        emb_ref=emb_ref, question=query
    )
    return prompt_template

def reRank(rerank, top_k, query, bm25_ref, faiss_ans):
    items = []
    max_length = 4000
    for doc, score in faiss_ans:
        items.append(doc)
    items.extend(bm25_ref)
    rerank_ref = rerank.predict(query, items)
    rerank_ref = rerank_ref[:top_k]
    # docs_sort = sorted(rerank_ref, key = lambda x:x.metadata["id"])
    emb_ref = ""
    for doc in rerank_ref:
        if len(emb_ref + doc.page_content) > max_length:
            break
        emb_ref = emb_ref + doc.page_content
    return emb_ref


if __name__ == "__main__":

    qwen7 = "./Qwen-7B-Chat"
    m3e = "./m3e-large"
    bge_reranker_large = "./bge-reranker-large"

    dp = DataProcess(pdf_path="./data/7215_slides.pdf")
    # @ pdf_parse.py
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)

    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)

    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)

    data = dp.data
    print("\nParsing and loading of PDF data is complete.\n")

    # Faiss召回
    faissretriever = FaissRetriever(m3e, data)
    print("\nfaissretriever load ok\n")

    # BM25召回
    bm25 = BM25(data)
    print("\nbm25 load ok\n")

    # LLM大模型
    llm = ChatLLM(qwen7)
    print("\nllm qwen load ok\n")

    # reRank模型
    rerank = reRankLLM(bge_reranker_large)
    print("\nrerank model load ok\n")

    # 对每一条测试问题，做答案生成处理
    start = time.time()
    with open("./data/7215_question.json", "r") as f:
        jdata = json.loads(f.read())
        print(f"\nTotal number of questions: {len(jdata)}\n")
        max_length = 4000
        for idx, line in enumerate(jdata):
            query = line["question"]

            # faiss召回topk
            faiss_context = faissretriever.GetTopK(query, 15)
            faiss_min_score = 0.0
            if len(faiss_context) > 0:
                faiss_min_score = faiss_context[0][1]
            cnt = 0
            emb_ref = ""
            for doc, score in faiss_context:
                cnt = cnt + 1
                # 最长选择max length
                if len(emb_ref + doc.page_content) > max_length:
                    break
                emb_ref = emb_ref + doc.page_content
                # 最多选择6个
                if cnt > 6:
                    break

            # bm2.5召回topk
            bm25_context = bm25.GetBM25TopK(query, 15)
            bm25_ref = ""
            cnt = 0
            for doc in bm25_context:
                cnt = cnt + 1
                if len(bm25_ref + doc.page_content) > max_length:
                    break
                bm25_ref = bm25_ref + doc.page_content
                if cnt > 6:
                    break

            # 构造合并bm25召回和向量召回的prompt
            emb_bm25_merge_inputs = get_emb_bm25_merge(
                faiss_context, bm25_context, query
            )

            # 构造bm25召回的prompt
            bm25_inputs = get_rerank(bm25_ref, query)

            # 构造向量召回的prompt
            emb_inputs = get_rerank(emb_ref, query)

            # rerank召回的候选，并按照相关性得分排序
            rerank_ref = reRank(rerank, 6, query, bm25_context, faiss_context)
            # 构造得到rerank后生成答案的prompt
            rerank_inputs = get_rerank(rerank_ref, query)

            batch_input = []
            batch_input.append(emb_bm25_merge_inputs)
            batch_input.append(bm25_inputs)
            batch_input.append(emb_inputs)
            batch_input.append(rerank_inputs)
            batch_input.append(query)
            # 执行batch推理
            batch_output = llm.infer(batch_input)
            line["answer_emb_bm25_merge"] = batch_output[0]  # 合并两路召回的结果
            line["answer_bm25"] = batch_output[1]  # bm召回的结果
            line["answer_emb"] = batch_output[2]  # 向量召回的结果
            line["answer_rerank"] = batch_output[3]  # 多路召回重排序后的结果
            line["answer_original"] = batch_output[4] # 原始模型
            line["answer_emb_ref"] = emb_ref
            line["answer_bm25_ref"] = bm25_ref
            line["answer_rerank_ref"] = rerank_ref
            # 如果faiss检索跟query的距离高于500，输出无答案
            if faiss_min_score > 500:
                line["answer_emb_ref"] = "无答案"
            else:
                line["answer_emb_ref"] = str(faiss_min_score)

        # 保存结果，生成submission文件
        json.dump(
            jdata,
            open("./data/7215_result.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        end = time.time()
        print(f"\nCost time of inference: {(end - start):.4f} seconds.\n")
