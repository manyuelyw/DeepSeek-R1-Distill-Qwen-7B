from huggingface_hub import hf_hub_download
import os

# 使用国内镜像站点
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载模型中的某个特定文件
file_path = hf_hub_download(
    repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
    filename="Qwen2.5-7B-Instruct-f16.gguf",
    local_dir=".",
    local_dir_use_symlinks=False  # 避免软链接问题
)

print(f"文件下载至：{file_path}")
