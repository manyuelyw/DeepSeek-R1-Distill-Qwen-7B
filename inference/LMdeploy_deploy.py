import time
import psutil
import os
import argparse
import torch
import matplotlib.pyplot as plt
import pynvml
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig


def get_gpu_memory(gpu_ids=None):
    try:
        result = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader').read()
        gpu_memory_list = [int(x) for x in result.strip().split('\n')]
        if gpu_ids is None:
            return gpu_memory_list
        return [gpu_memory_list[i] for i in gpu_ids]
    except Exception as e:
        print(f"[ERROR] 获取GPU显存失败: {e}")
        return [0] * (len(gpu_ids) if gpu_ids else 1)


def get_peak_gpu_memory(gpu_ids):
    pynvml.nvmlInit()
    peak_list = []
    for i in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        peak_list.append(mem_info.used // 1024 // 1024)  # MB
    return peak_list


def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def plot_metrics(latencies, throughputs, output_path):
    rounds = list(range(1, len(latencies) + 1))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, throughputs, marker='o')
    plt.title("Throughput per Round (tokens/s)")
    plt.xlabel("Round")
    plt.ylabel("Throughput")

    plt.subplot(1, 2, 2)
    plt.plot(rounds, latencies, marker='x', color='orange')
    plt.title("Latency per Round (s)")
    plt.xlabel("Round")
    plt.ylabel("Latency")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] 图表已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--gpu', type=str, default='0', help='使用的GPU ID，例如: 0 或 0,1')
    parser.add_argument('--runs', type=int, default=5, help='测试轮数')
    parser.add_argument('--output', type=str, default='lmdeploy_results.txt', help='结果保存路径')
    parser.add_argument('--plot', type=str, default='lmdeploy_plot.png', help='图表保存路径')
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu.split(',')]
    prompts = [
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "John writes 20 pages a day.  How long will it take him to write 3 books that are 400 pages each?",
        "Ed has 2 dogs, 3 cats and twice as many fish as cats and dogs combined. How many pets does Ed have in total?",
        "James buys 5 packs of beef that are 4 pounds each.  The price of beef is $5.50 per pound.  How much did he pay?",
        "A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?"
    ]

    print(f"[INFO] 正在加载模型: {args.model}")
    ram_before_load = get_ram_usage()
    gpu_before_load = get_gpu_memory(gpu_ids)

    backend_config = TurbomindEngineConfig(tp=1, devices=gpu_ids)
    gen_config = GenerationConfig(top_p=0.8, top_k=40, temperature=0.8, max_new_tokens=2048, do_sample=True)
    start_load = time.time()
    chatbot = pipeline(args.model, backend_config=backend_config)
    load_time = time.time() - start_load

    ram_after_load = get_ram_usage()
    gpu_after_load = get_gpu_memory(gpu_ids)
    print(f"[INFO] 模型加载完成，耗时 {load_time:.2f}s")

    print("[INFO] 模型预热中...")
    _ = chatbot(prompts[0], gen_config=gen_config)

    print("[INFO] 开始性能测试...")
    latencies, throughputs = [], []
    total_tokens, total_time = 0, 0

    ram_before_infer = get_ram_usage()
    gpu_before_infer = get_gpu_memory(gpu_ids)

    for i in range(args.runs):
        round_start = time.time()
        generated_tokens = 0

        for prompt in prompts:
            output = chatbot(prompt, gen_config=gen_config)
            token_count = len(output.token_ids)
            generated_tokens += token_count

        round_time = time.time() - round_start
        latencies.append(round_time)
        throughputs.append(generated_tokens / round_time)
        total_tokens += generated_tokens
        total_time += round_time

        print(f"[Round {i+1}] 耗时: {round_time:.2f}s, Tokens: {generated_tokens}, 吞吐率: {generated_tokens / round_time:.2f} tokens/s")

    ram_after_infer = get_ram_usage()
    gpu_after_infer = get_gpu_memory(gpu_ids)
    peak_gpu_mem = get_peak_gpu_memory(gpu_ids)

    avg_latency = sum(latencies) / (args.runs * len(prompts))
    avg_throughput = total_tokens / total_time

    print(f"\n[SUMMARY]")
    print(f"平均延迟: {avg_latency:.2f} 秒/请求")
    print(f"平均吞吐率: {avg_throughput:.2f} tokens/s")
    print(f"加载阶段RAM增加: {ram_after_load - ram_before_load:.2f} MB")
    print(f"推理阶段RAM增加: {ram_after_infer - ram_before_infer:.2f} MB")
    print(f"加载阶段GPU显存增加: {[after - before for after, before in zip(gpu_after_load, gpu_before_load)]} MB")
    print(f"推理阶段GPU显存变化: {[after - before for after, before in zip(gpu_after_infer, gpu_before_infer)]} MB")
    print(f"峰值GPU显存使用: {peak_gpu_mem} MB")

    with open(args.output, 'w') as f:
        f.write(f"模型: {args.model}\n")
        f.write(f"平均延迟: {avg_latency:.2f} 秒/请求\n")
        f.write(f"平均吞吐率: {avg_throughput:.2f} tokens/s\n")
        f.write(f"加载阶段RAM变化: {ram_after_load - ram_before_load:.2f} MB\n")
        f.write(f"推理阶段RAM变化: {ram_after_infer - ram_before_infer:.2f} MB\n")
        f.write(f"加载阶段GPU变化: {[after - before for after, before in zip(gpu_after_load, gpu_before_load)]} MB\n")
        f.write(f"推理阶段GPU变化: {[after - before for after, before in zip(gpu_after_infer, gpu_before_infer)]} MB\n")
        f.write(f"峰值GPU显存使用: {peak_gpu_mem} MB\n")

    plot_metrics(latencies, throughputs, args.plot)


if __name__ == '__main__':
    main()
