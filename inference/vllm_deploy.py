from vllm import LLM, SamplingParams
import time
import torch
import psutil
import os


# 记录显存占用
def get_gpu_memory(gpu_id=0):
    """获取指定GPU的显存使用情况"""
    try:
        result = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader').read()
        # 按行分割输出
        gpu_memory_list = result.strip().split('\n')

        # 确保有对应GPU的信息
        if gpu_id < len(gpu_memory_list):
            return int(gpu_memory_list[gpu_id])
        else:
            print(f"警告：请求的GPU ID {gpu_id} 不存在，系统中只有 {len(gpu_memory_list)} 个GPU")
            return 0
    except Exception as e:
        print(f"获取GPU显存时出错: {e}")
        return 0


# 记录内存占用
def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


# 模型路径
model_path = "./Models/qwen/Qwen2.5-7B-Instruct"

# 初始化模型
print("加载模型中...")
start_time = time.time()
llm = LLM(model=model_path, dtype="float16")
load_time = time.time() - start_time
print(f"模型加载完成，耗时: {load_time:.2f}秒")

# 记录模型加载后的显存占用
gpu_mem_after_load = get_gpu_memory()
print(f"模型加载后显存占用: {gpu_mem_after_load} MB")

# 示例提示(中文)
# prompts = [
#     "介绍一下量子计算的基本原理",
#     "简述人工智能的发展历程",
#     "解释一下区块链技术",
#     "什么是自然语言处理？",
#     "请介绍机器学习中的监督学习和无监督学习"
# ]

# 示例提示(英文)
prompts = [
    "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
    "John writes 20 pages a day.  How long will it take him to write 3 books that are 400 pages each?",
    "Ed has 2 dogs, 3 cats and twice as many fish as cats and dogs combined. How many pets does Ed have in total?",
    "James buys 5 packs of beef that are 4 pounds each.  The price of beef is $5.50 per pound.  How much did he pay?？",
    "A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?"
]

# 采样参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2048)

# 预热
print("模型预热中...")
llm.generate(prompts[0], sampling_params)

# 性能测试
print("开始性能测试...")
ram_before = get_ram_usage()
gpu_before = get_gpu_memory()

num_runs = 10
total_tokens = 0
total_time = 0

for i in range(num_runs):
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()

    # 计算生成的总token数
    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_tokens += generated_tokens

    # 计算单次运行时间
    run_time = end - start
    total_time += run_time

    print(f"运行 {i + 1}/{num_runs}: {run_time:.2f}秒, {generated_tokens} tokens, "
          f"吞吐率: {generated_tokens / run_time:.2f} tokens/秒")

# 计算平均性能指标
avg_latency = total_time / (num_runs * len(prompts))
avg_throughput = total_tokens / total_time
ram_after = get_ram_usage()
gpu_after = get_gpu_memory()

print(f"\n平均延迟: {avg_latency:.2f}秒/请求")
print(f"平均吞吐率: {avg_throughput:.2f} tokens/秒")
print(f"RAM使用变化: {ram_after - ram_before:.2f} MB")
print(f"GPU显存使用变化: {gpu_after - gpu_before} MB")
print(f"峰值GPU显存占用: {gpu_mem_after_load} MB")

# 保存结果
with open("vllm_results.txt", "w") as f:
    f.write(f"模型: Qwen2.5-7B-Instruct\n")
    f.write(f"框架: vllm\n")
    f.write(f"平均延迟: {avg_latency:.2f}秒/请求\n")
    f.write(f"平均吞吐率: {avg_throughput:.2f} tokens/秒\n")
    f.write(f"峰值GPU显存占用: {gpu_mem_after_load} MB\n")