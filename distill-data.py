import requests
import json
import time
import os
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class DeepSeekAPIWrapper:
    """DeepSeek API包装器，提供安全高效的模型调用"""

    def __init__(self, api_key=None, model="deepseek-reasoner", max_workers=5):
        # 优先从环境变量获取API密钥
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("请设置DeepSeek API密钥，通过环境变量或构造函数传入")

        self.model = model
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.max_workers = max_workers

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _single_request(self, prompt, temperature=0.7, max_tokens=512):
        """单个API请求，带重试机制"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))

        # 处理常见错误
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 429:
            logger.warning("请求频率过高，等待后重试")
            time.sleep(5)  # 等待5秒后重试
            raise Exception("Rate limit exceeded")
        elif response.status_code in [500, 502, 503, 504]:
            logger.warning(f"服务器错误 ({response.status_code})，等待后重试")
            time.sleep(3)
            raise Exception(f"Server error: {response.status_code}")
        else:
            error_msg = response.json().get("error", {}).get("message", "未知错误")
            logger.error(f"API请求失败: {response.status_code} - {error_msg}")
            raise Exception(f"API error: {response.status_code}")

    def generate_responses(self, prompts, temperature=0.7, max_tokens=512, batch_size=10):
        """批量生成响应，使用线程池提高效率"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for prompt in prompts:
                future = executor.submit(
                    self._single_request,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                futures.append(future)

            # 使用tqdm显示进度
            for future in tqdm(futures, desc="调用API"):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"生成响应时出错: {str(e)}")
                    results.append(None)  # 出错时添加None，保持结果长度一致

        return results


def load_data(file_path):
    """安全加载jsonl格式数据"""
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        return []


def process_dataset(api_wrapper, raw_data, output_path, max_examples=None, temperature=0.7):
    """处理数据集并保存结果，支持断点续传"""
    # 检查是否有已处理的部分
    processed_data = []
    if os.path.exists(output_path):
        processed_data = load_data(output_path)
        logger.info(f"发现已处理数据，加载了 {len(processed_data)} 条记录")

    # 计算需要处理的数据
    total = len(raw_data) if max_examples is None else min(max_examples, len(raw_data))
    remaining_data = raw_data[len(processed_data):total]

    if not remaining_data:
        logger.info("没有需要处理的新数据")
        return processed_data

    logger.info(f"准备处理 {len(remaining_data)} 条新数据")

    # 提取所有prompt
    prompts = [item["prompt"] for item in remaining_data]

    # 调用API生成响应
    responses = api_wrapper.generate_responses(
        prompts,
        temperature=temperature,
        max_tokens=2048
    )

    # 合并结果
    new_data = []
    for i, (item, response) in enumerate(zip(remaining_data, responses)):
        if response is not None:
            new_data.append({
                "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
                "input": item["instruction"],
                "output": response
            })
        else:
            logger.warning(f"第 {i + 1} 条数据生成失败，跳过")

    # 合并新旧数据并保存
    all_data = processed_data + new_data
    save_dataset(all_data, output_path)

    return all_data


def save_dataset(dataset, output_path):
    """保存数据集，确保中文正确编码"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"成功保存 {len(dataset)} 条数据到 {output_path}")
    except Exception as e:
        logger.error(f"保存数据失败: {str(e)}")


if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = "./instruction-data/gsm8k/train.jsonl"
    OUTPUT_FILE = "./raw_dataset.jsonl"
    MAX_EXAMPLES = 1
    TEMPERATURE = 0.7  # 蒸馏时使用较高温度

    # 初始化API包装器
    api_wrapper = DeepSeekAPIWrapper(
        api_key="sk-68725f97ab9547cbba862f0d0076ba44",
        model="deepseek-reasoner",
        max_workers=1  # 并发请求数
    )

    # 加载原始数据
    raw_data = load_data(INPUT_FILE)
    if not raw_data:
        logger.error("没有数据可处理，程序退出")
    else:
        logger.info(f"已加载 {len(raw_data)} 条原始数据")

    # 处理数据
    process_dataset(
        api_wrapper=api_wrapper,
        raw_data=raw_data,
        output_path=OUTPUT_FILE,
        max_examples=MAX_EXAMPLES,
        temperature=TEMPERATURE
    )
