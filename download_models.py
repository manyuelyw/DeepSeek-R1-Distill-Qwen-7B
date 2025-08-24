from huggingface_hub import snapshot_download
import os

# 设置国内镜像站点
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载模型（支持断点续传）
snapshot_download(
    repo_id="Qwen/Qwen2.5-32B-Instruct",
    local_dir="../shareLLMs/LLMs/model/Qwen2.5-32B-Instruct",
    resume_download=True
)
