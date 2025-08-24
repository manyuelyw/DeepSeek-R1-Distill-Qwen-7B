import json
import re
import os
import argparse


def extract_answer_from_boxed(text):
    """从文本中提取\boxed{}内的内容作为答案"""
    if not text:
        return None

    # 使用正则表达式匹配\boxed{...}模式
    # 匹配模式：\boxed{ followed by any characters (non-greedy) followed by }
    match = re.search(r'\\boxed\{(.*?)\}', text)

    if match:
        return match.group(1).strip()  # 返回括号内的内容并去除首尾空格
    return None


def process_jsonl_file(raw_file, label_file, output_file):
    """处理JSONL文件，提取答案并保存"""
    results = []

    # 检查文件是否存在
    if not os.path.exists(raw_file):
        print(f"错误：原始文件 {raw_file} 不存在")
        return False
    if not os.path.exists(raw_file):
        print(f"错误：目标文件 {label_file} 不存在")
        return False

    try:
        # 读取并处理JSONL文件
        with open(raw_file, 'r', encoding='utf-8') as f_raw:
            with open(label_file, 'r', encoding='utf-8') as f_label:
                for line_raw, line_label in zip(f_raw, f_label):
                    # 解析JSON
                    data_raw = json.loads(line_raw.strip())
                    data_label = json.loads(line_label.strip())

                    # 检查output字段是否存在
                    if data_raw['output'] == '' or data_label['output'] == '':
                        continue
                    answer_raw = extract_answer_from_boxed(data_raw['output'])
                    answer_label = extract_answer_from_boxed(data_label['output'])
                    if answer_raw == answer_label:
                        # 保存这条deepseek蒸馏数据
                        results.append({
                            'original': data_raw,
                            'extracted_answer': answer_raw
                        })

        # 输出结果
        if results:
            print(f"deepseek蒸馏数据中答案正确的有 {len(results)} 条")

            # 如果指定了输出文件，则保存结果
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in results:
                        # 只保存原始数据和提取的答案
                        simplified_item = {
                            'instruction': item['original'].get('instruction', ''),
                            'input': item['original'].get('input', ''),
                            'output': item['original'].get('output', '')
                        }
                        f.write(json.dumps(simplified_item, ensure_ascii=False) + '\n')
                print(f"结果已保存到 {output_file}")
        return True

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return False


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_file', type=str, default="./instruction-data/gsm8k/raw_data.jsonl")
    parser.add_argument('--label_file', type=str, default="./instruction-data/gsm8k/test.jsonl")
    parser.add_argument('--output_file', type=str, default="./instruction-data/gsm8k/final_train.jsonl")

    args = parser.parse_args()

    process_jsonl_file(args.raw_file, args.label_file, args.output_file)


if __name__ == "__main__":
    main()
