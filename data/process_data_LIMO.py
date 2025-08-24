import json


def load_data(file_path):
    """安全加载jsonl格式数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_dataset(dataset, output_path):
    """保存数据集，确保中文正确编码"""
    new_data = []
    for item in dataset:
        new_data.append({
            "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
            "input": item["question"],
            "output": item["solution"]
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = "limo_dataset.json"
    OUTPUT_FILE = "process_dataset_LIMO.json"

    # 加载原始数据
    raw_data = load_data(INPUT_FILE)
    save_dataset(raw_data, OUTPUT_FILE)
