from huggingface_hub import snapshot_download

# 首先创建.env并写入HF_TOKEN=YOUR_TOKEN
# 随后pip install python-dotenv

from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_TOKEN")

"""
snapshot_download(
        repo_id="moka-ai/m3e-large",
        token=token,
        local_dir="./pre_train_model/m3e-large/"
)

snapshot_download(
        repo_id="Qwen/Qwen-7B-Chat",
        token=token,
        local_dir="./pre_train_model/Qwen-7B-Chat/"
)

snapshot_download(
    repo_id="shibing624/text2vec-base-chinese",
    token=token,
    local_dir="./pre_train_model/text2vec-base-chinese/",
)
"""

# snapshot_download(
#     repo_id="THUDM/chatglm2-6b",
#     token=token,
#     local_dir="/scratch/project_2006362/v/.v_scratchs/llm/20_financial_AI_expert/proj_2/data/pretrained_models/chatglm2-6b/",
# )

snapshot_download(
    repo_id="Qwen/Qwen1.5-7B",
    token=token,
    local_dir="/root/autodl-tmp/.v_scratchs/llm/15_lynk_RAG_proj/Qwen-7B-Chat",
)