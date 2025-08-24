import requests
import time
import psutil
import os
import pynvml


def get_gpu_memory(gpu_id=0):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return meminfo.used / 1024 / 1024  # MB
    except Exception as e:
        print(f"获取GPU显存时出错: {e}")
        return 0


def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def monitor_peak_gpu(gpu_id=0, interval=0.5, duration=30):
    peak = 0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        for _ in range(int(duration / interval)):
            used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
            peak = max(peak, used)
            time.sleep(interval)
    except Exception as e:
        print(f"监控GPU显存时出错: {e}")
    return peak


print("启动Ollama服务...")
os.system("ollama serve &")
time.sleep(10)

prompts = [
    "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
    "John writes 20 pages a day.  How long will it take him to write 3 books that are 400 pages each?",
    "Ed has 2 dogs, 3 cats and twice as many fish as cats and dogs combined. How many pets does Ed have in total?",
    "James buys 5 packs of beef that are 4 pounds each.  The price of beef is $5.50 per pound.  How much did he pay?",
    "A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?"
]

print("记录加载前内存和显存...")
ram_before_load = get_ram_usage()
gpu_before_load = get_gpu_memory()

print("等待模型加载...")
peak_gpu_during_load = monitor_peak_gpu(gpu_id=0, duration=30)

print("模型加载完成")
ram_after_load = get_ram_usage()
gpu_after_load = get_gpu_memory()

print("模型预热中...")
requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "Qwen2.5-7b-Instruct",
        "prompt": prompts[0],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2048
        }
    }
)

print("开始性能测试...")
ram_before_test = get_ram_usage()
gpu_before_test = get_gpu_memory()

num_runs = 10
total_tokens = 0
total_time = 0
peak_gpu_during_test = 0

for i in range(num_runs):
    start = time.time()
    generated_tokens = 0

    for prompt in prompts:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "Qwen2.5-7b-Instruct",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 2048
                }
            }
        )
        if response.status_code == 200:
            data = response.json()
            generated_tokens += len(data.get("response", "").split())

    run_time = time.time() - start
    total_tokens += generated_tokens
    total_time += run_time
    print(f"运行 {i + 1}/{num_runs}: {run_time:.2f}秒, {generated_tokens} tokens, 吞吐率: {generated_tokens / run_time:.2f} tokens/秒")

    peak_now = get_gpu_memory()
    peak_gpu_during_test = max(peak_gpu_during_test, peak_now)

ram_after_test = get_ram_usage()
gpu_after_test = get_gpu_memory()

avg_latency = total_time / (num_runs * len(prompts))
avg_throughput = total_tokens / total_time

print(f"\n平均延迟: {avg_latency:.2f}秒/请求")
print(f"平均吞吐率: {avg_throughput:.2f} tokens/秒")
print(f"RAM加载增加: {ram_after_load - ram_before_load:.2f} MB")
print(f"RAM推理增加: {ram_after_test - ram_before_test:.2f} MB")
print(f"GPU加载增加: {gpu_after_load - gpu_before_load:.2f} MB")
print(f"GPU推理增加: {gpu_after_test - gpu_before_test:.2f} MB")
print(f"峰值GPU显存（加载期间）: {peak_gpu_during_load:.2f} MB")
print(f"峰值GPU显存（推理期间）: {peak_gpu_during_test:.2f} MB")

with open("ollama_results.txt", "w") as f:
    f.write("模型: Qwen2.5-7B-Instruct\n")
    f.write("框架: ollama\n")
    f.write(f"平均延迟: {avg_latency:.2f}秒/请求\n")
    f.write(f"平均吞吐率: {avg_throughput:.2f} tokens/秒\n")
    f.write(f"RAM加载增加: {ram_after_load - ram_before_load:.2f} MB\n")
    f.write(f"RAM推理增加: {ram_after_test - ram_before_test:.2f} MB\n")
    f.write(f"GPU加载增加: {gpu_after_load - gpu_before_load:.2f} MB\n")
    f.write(f"GPU推理增加: {gpu_after_test - gpu_before_test:.2f} MB\n")
    f.write(f"峰值GPU显存（加载期间）: {peak_gpu_during_load:.2f} MB\n")
    f.write(f"峰值GPU显存（推理期间）: {peak_gpu_during_test:.2f} MB\n")

os.system("pkill ollama")
