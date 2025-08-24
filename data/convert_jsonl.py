import json


def convert_to_jsonl(input_path, output_path):
    # 读取 JSON 文件（数组格式）
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 写入 JSONL 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"✅ 转换完成，已保存为 JSON 文件：{output_path}")


# 使用示例
if __name__ == "__main__":
    input_file = './instruction-data/alphaca/alpaca_en_demo.json'       # 原始文件路径
    output_file = './instruction-data/alphaca/final_data.json'     # 输出 JSON 文件路径
    convert_to_jsonl(input_file, output_file)
