import mteb
from sentence_transformers import SentenceTransformer

# export HF_DATASETS_CACHE=/root/autodl-tmp/embedding_evaluation/datasets/

# Define the sentence-transformers model name
model_names = ["/root/autodl-tmp/embedding_evaluation/gte-Qwen2-1.5B-instruct", "/root/autodl-tmp/.v_scratchs/llm/15_lynk_RAG_proj/m3e-large"]

for i, model_name in enumerate(model_names):
    model = mteb.get_model(model_name) # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)
    tasks = [
        mteb.get_task("AmazonReviewsClassification", languages = ["eng", "fra"]),
        mteb.get_task("Banking77Classification"),
        # mteb.get_task("AppsRetrieval"),
        mteb.get_task("AFQMC")
    ]

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=f"results/m3e", verbosity=2) if i == 0 else evaluation.run(model, output_folder=f"results/gte-qwen", encode_kwargs={"batch_size": 4}, verbosity=2)
