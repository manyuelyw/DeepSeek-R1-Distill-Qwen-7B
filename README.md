# 从deepseek-r1中蒸馏数据流程：
首先，从deepseek官网进入API调用平台，充值，创建API Key；
然后是数据集选择，选择数学推理领域常用开源数据集gsm8k作为指令数据集，考虑到gsm8k数据集为小学数学推理任务，常用于评估 Chain-of-Thought 推理能力，数据格式如下：
{
    "instruction": "Tim buys 3 dozen eggs.  Eggs cost $.50 each.  How much did he pay for eggs?", 
    "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\nTim buys 3 dozen eggs.  Eggs cost $.50 each.  How much did he pay for eggs?<|im_end|>\n<|im_start|>assistant\n", 
    "input": "", 
    "output": "He bought 3*12=<<3*12=36>>36 eggs\nSo they cost 36*.5=$<<36*.5=18>>18\nSo the answer is \\(\\boxed{18}\\)."
}

然后，使用deepseek-r1-0528作为教师模型，喂入上述指令数据集，调用API生成数据，使用 DeepSeek-R1 生成 对问题的完整推理过程cot：（改进效率的优化版在1.2，见distill_data.py）
import requests
import json

# prompt前缀设计为：Please reason step by step, and put your final answer within \\boxed{}.
def generate_teacher_response(prompt):
    """调用DeepSeek API生成数据"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": "API-KEY"    # 输入获取的API-KEY
    }
    payload = {
        "model": "deepseek-reasoner",    # 调用DeepSeek-R1-0528
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,  # 控制随机性，蒸馏时建议设为2-5
        "max_tokens": 1024    # 经实验发现设置为512时生成的output中很多都还没得到结果
    }
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )
    return response.json()["choices"][0]["message"]["content"]

# 批量生成数据
dataset = []
for example in raw_data:
    teacher_output = generate_teacher_response(example["instruction"])
    dataset.append({
        "instruction": example["instruction"],
        "teacher_response": teacher_output
    })

蒸馏数据预处理：获取deepseek-r1的输出之后，要将得到的数据集进行格式转换（数据集的格式修改为符合Qwen的模板要求）和正确性过滤（仅保留最后推理结果正确的deepseek蒸馏出的cot数据）。原因一是微调时输入输出的格式是否与模型原始的指令遵循格式不一致可能导致模型混淆；二是从deepseek-R1的API中蒸馏数据设置的max_token=1024（2048也可以跑代价太高昂）会有cot生成过程被截断，还没有生成到答案，而如果数据本身存在模糊、错误或不一致，模型后面微调学到的就是错误的东西。（处理代码略，见fliter_right.py文件）经过上述处理后最终得到的数据集gsm8k-deepseek-r1-200的格式如下：(过滤后共221条)
{
  "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
  "input": "question",
  "output": "cot"
}

1.2 遇到的错误及解决方法整理：
●已解决问题：当调用Deepseek-R1的API提取输出的时候，代码虽未报错，但效率很低，平均40s生成一条回复（1024token截断），因此后来对构建API wrapper并发请求，生成速率有了显著提升，其中，关键部分代码如下：
# 初始化API包装器
api_wrapper = DeepSeekAPIWrapper(
    api_key="sk-68725f97ab9547cbba862f0d0076ba44",
    model="deepseek-reasoner",
    max_workers=5  # 并发请求数
)
...
# 调用API生成响应
responses = api_wrapper.generate_responses(
    prompts,
    temperature=temperature,
    max_tokens=2048
)
...
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

●未解决问题：但是在调用API生成回复过程中，观察生成数据，当并发请求数越大时，出现生成数据为空或严重不全的情况就会越多。（考虑可能是请求响应时间设置方面的问题？暂未解决。。）
1.3 学习过程中其他收获：
●通过API获取deepseek输出数据过程中，一些错误码及原因：

6.10~6.13微调训练
2.1 实验环境搭建：
使用本地pycharm（专业版）连接远程服务器主机以实现远程代码的本地编写与调试；具体流程截图如下：
配置Deployment：

使用远程python解释器：



文件映射关系设置：

硬件环境：GPU使用单卡24GB显存的4090（整个微调训练流程中单卡最大所需运行内存23GB）。创建虚拟conda环境：
# 创建独立的conda环境deepseek并激活
conda create -n deepseek python=3.10
conda activate deepseek

# 安装torch
pip3 install torch torchvision torchaudio

2.1 微调流程：
使用llama-factory框架实现大模型的高效微调（lora）
# 首先，拉取LLaMA-Factory项目到远程服务器
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

# 安装运行LLaMA-Factory所需环境
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

使用1中从deepseek-r1中得到的蒸馏数据集gsm8k-deepseek-r1-200直接微调Qwen2.5-7B-Instruct。将上述数据集加入llama-factory/data文件夹下并修改llama-factory代码中的dataset_info：
"mydataset_gsm8k_deepseek_right_qwen2.5_7B": {
"file_name": "mydataset_gsm8k_deepseek_right_qwen2.5_7B.json"
},

准备学生模型Qwen2.5-7B-Instruct： 使用以下命令从 Hugging Face 下载模型文件，并支持断点续传和使用镜像源加速下载（Hugging Face 官方站点 https://huggingface.co 有时在国内访问速度较慢甚至会超时）。
# 设置环境变量HF_ENDPOINT，使用国内镜像站（hf-mirror.com）来访问模型下载服务。
export HF_ENDPOINT=https://hf-mirror.com

# 通过 Hugging Face CLI 下载指定模型，支持断点续传功能
huggingface-cli download --resume-download Qwen/Qwen2.5-7B-Instruct --local-dir ./Models/qwen/Qwen2.5-7B-Instruct

中途有时候会出现下载不全，连接断掉的问题，因此后面为了求更加稳定快速的下载方式，使用国内的modelscope社区下载：（首先pip install modelscope）
下载整个文件夹：
modelscope download --model second-state/Qwen2.5-7B-Instruct-GGUF

下载文件夹下指定的单个文件：
modelscope download --model second-state/Qwen2.5-7B-Instruct-GGUF README.md --local_dir ./dir

模型成功开始下载截图：

由于蒸馏数据量只有1200多条，因此需要重点关注过拟合问题、训练效率优化和参数敏感性调整。因此设计微调训练参数如下：
总批次大小不宜过大，否则每个epoch的更新次数太少：batch_size = 2；
小数据量下epoch增加但不能过大，由于在少量数据上训练过多轮次，必然导致严重的过拟合：epoch = 5；
学习率降低：learning_rate = 5.0e-6(1.0e-5 )；
适当降低lora微调引入的矩阵的维度，更小的秩进一步减少了投入训练的模型参数量，从而让模型的表达能力适r度受限，能起到一定的正则化效果，避免在小数据集上过拟合。lora_rank: 8；
其中高效微调lora的原理和核心思想如下：

每30step进行一次验证集效果评估，观察验证集损失有无稳定下降，若出现上升，则及时停止训练（使用早停策略）；因此，训练参数文件总设置如下：
### model
model_name_or_path: ../Models/qwen/Qwen2.5-7B-Instruct

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8 # 降低秩，减少过拟合风险
lora_alpha: 16
lora_dropout: 0.1 # 新增：LORA层 dropout，防止过拟合

### dataset
dataset: mydataset_gsm8k_deepseek_right_qwen2.5_7B
template: qwen
cutoff_len: 2048
max_samples: 221 # 严格匹配数据量
overwrite_cache: true
preprocessing_num_workers: 8 # 小数据适当降低，节省资源

### output
output_dir: ../Models/qwen/lora_sft_right_Qwen_7B_lr_constant
logging_steps: 5 # 更频繁监控训练
save_steps: 50 # 小步长保存，便于选择最佳模型
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 1
optim: adamw_torch
learning_rate: 1.0e-5 # 降低学习率，避免参数震荡
num_train_epochs: 5 # 增加epoch数（小数据需更多轮次学习模式，但需防过拟合）
lr_scheduler_type: constant # 固定学习率，避免过早衰减
warmup_ratio: 0.05 # 缩短热身阶段，快速进入有效训练
bf16: true
ddp_timeout: 180000000
weight_decay: 0.01 # 新增：权重衰减，抑制过拟合

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50 # 每50步验证一次，及时发现过拟合

后面考虑到调用API的成本问题:

&&小规模数据集下模型蒸馏微调效果不好等问题，我将现有的蒸馏数据集gsm8k-deepseek-r1-200融合开源的从deepseek-r1中蒸馏得到的数据集Alphaca、LIMO等，得到最终的训练weitiao样本，用于微调Qwen2.5-7B-Instruct。
然后运行训练脚本，进行微调：
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/myllama3_lora_sft_gsm8k_deepseek_right_qwen2.5_7B.yaml

跟踪微调过程分析：
每50个step保存一次checkpoint，最后根据训练损失下降情况和验证集损失情况确定最优checkpoint


合并lora-ckpt和原模型得到完整的微调后的模型：
首先编写模型合并脚本qwen_lora_pretrain_merge.yaml
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters

### model
model_name_or_path: ../TokenSkip-main/model/qwen/Qwen2.5-7B-Instruct
adapter_name_or_path: ../TokenSkip-main/model/qwen/lora_sft_llmlingua2_Qwen_7B_lr_5e-5
template: qwen
trust_remote_code: true

### export
export_dir: ../TokenSkip-main/model/qwen/Qwen2.5-7B_lora_pretrain_merge
export_size: 5
export_device: cpu
export_legacy_format: false

然后运行合并脚本：
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml

●构建评测数据集
数学推理类：
复杂数学推理任务：AIME(2024)、AMC23
一般数学推理任务（难度不等）：GSM8K、Math、svamp、asdiv、mawps、carp_en、tabmwp、minerva_math、GaoKao2023En、OlympiadBench、College Math
多项选择数学问题：aqua、sat_math、mmlu_stem
通识理解类：
通识知识理解：MMLU、MMLU_Pro
●跑模型评测

评测数据集与相应的评测脚本是代码的evaluation_benchmark部分，如下图所示：

跑数学推理类评测基准：
首先编写run_eval_qwen2_math.sh，指定待评测模型：
# Evaluate Qwen2.5-Math-Instruct
PROMPT_TYPE="qwen25-math-cot"

## Qwen2.5-7B-instruct
export CUDA_VISIBLE_DEVICES="1,2,3,4"
MODEL_NAME_OR_PATH="../../Models/qwen/Qwen2.5-7B_lora_pretrain_merge_Alphaca"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

然后跑脚本：
cd evaluation_benchmark/evaluation

sh sh/run_eval_qwen2_math.sh

跑通识理解类评测基准：



●微调前后效果对比

观察发现，复杂数学推理任务基准上指标结果变化不大；一般数学推理任务上性能仅有细微提升，也有些任务上（特别是通识知识理解任务mmlu）出现反而有降低的情况，因此通过查阅资料与观察分析才知道这种情况其实很常见，尤其是在小数据集微调大模型时。性能下降的主要原因分析：
1.过拟合了
在较小的数据集上训练，即使只训练几个epoch，模型也很快会记住这些具体的样本，而不是学到通用的解题方法。这表现为模型可能在这200条微调训练集上表现很好（甚至完美），但在未见过的GSM8K测试题或其他类型问题上表现很差。它失去了泛化能力。
2.灾难性遗忘
这是最可能的原因。预训练和指令微调阶段，模型吸收了海量的通用知识和技能（包括语言理解、常识、推理模式等）。当用仅200条GSM8K数据进行微调时，模型会过度专注于学习这200条特定数学题的解题模式。因此为了拟合这个小数据集，模型可能会“牺牲”或覆盖掉之前学习到的、与这200条数据不完全一致但更广泛的通识知识和通用推理能力。这就像为了记住几棵特定的树而忘记了一片森林的样子。在这种情况下，数学推理能力可能没提升多少（甚至下降，见2），但通识理解和其他领域的推理能力显著下降。
3.1221条数据太少
对于拥有7B参数的庞大模型来说，200条样本的信息量微乎其微。模型需要大量的、多样化的数据来稳定地学习到泛化能力强的模式。另外，GSM8K 数据本身的局限性使得即使是完整的GSM8K（grade school math）数据集，也主要聚焦于小学水平的数学应用题（虽然对模型来说已经很有挑战性），其覆盖的问题类型、解题步骤、语言表达、涉及的背景知识的多样性，很可能远低于模型原始训练数据（包含海量网页、书籍、百科等）。这导致模型无法从这小样本中学到稳健的数学推理能力，反而更容易学到数据集中的噪声或特定偏见，导致数学推理能力本身也没提升甚至下降。同时，由于数据单一，模型其他能力得不到“复习”，也退化了。
6.16~6.18分析各大大模型推理加速框架，vllm,deploy,ollama等优缺点
当前主流大语言模型推理部署框架：vLLM、LMDeploy、Ollama等。它们分别代表了高性能推理、高效量化和轻量部署的三种技术路线。vLLM 在吞吐量和多卡支持上占据优势，LMDeploy 通过量化技术突破硬件限制，Ollama 则以极简设计降低使用门槛。
3.1 vllm（UC Berkeley）
●简介：vllm 是一款高性能的 LLM 推理和服务引擎，专为大模型优化。它采用 PagedAttention 技术，有效减少 KV 缓存的内存碎片，提高内存利用率和推理速度。
●核心技术：
○PagedAttention：借鉴操作系统分页机制，将 KV 缓存分块存储，有效管理长上下文（8192 tokens），显存利用率提升至 96% 以上，内存浪费低于 4%。
○连续批处理（Continuous Batching）：动态调度请求，减少 GPU 空闲时间，吞吐量比 Hugging Face Transformers 高 8.5-24 倍。
○多 GPU 支持：张量并行（Tensor Parallelism）和流水线并行（Pipeline Parallelism），支持多卡分布式推理。
○项目地址：https://github.com/vllm-project/vllm
●优势：
○支持多用户、多请求并发推理，因此吞吐量很高、具有超强推理性能，
○基于 PagedAttention 机制节省了 KV Cache 显存，硬件资源利用更充分
○兼容 OpenAI API 风格（/v1/completions, /v1/chat/completions）
○支持多模态输入（文本、图像、音频）
●劣势：
○支持的部署场景较为固定（Web API 服务化），部署配置相对复杂，依赖 CUDA 环境；
○不支持非 Transformer-based 架构；
○模型格式需转换，部分闭源模型（如 GPT）不支持。
●适用场景：
适合高吞吐、多用户多请求的高并发场景的推理，如高并发的生产环境——企业级智能客服、AI 助手等；适合Chat 服务部署（如 ChatGPT clone），LLM 服务化。
3.2 Ollama
●简介：Ollama 是一个简单易用的 LLM 部署工具，提供了简洁的 API 和 CLI 界面。它支持多种量化方法，能快速部署模型并进行推理。
●核心技术：
基于 llama.cp 开发，支持 CPU 推理，安装简单，开箱即用，适合快速原型开发和测试
○思考模式（Thinking Mode）：生成推理路径，提升可解释性。
○流式工具调用：支持实时触发外部工具，增强交互体验。
○内存优化：自动切换 CPU/GPU，INT4 量化模型显存占用低至 4.7GB。
○项目地址：https://github.com/ollama/ollama
●优势：
○本地化部署简洁，安装简单：brew install ollama；支持 1700 + 预训练模型一键运行，适合快速原型开发。
○自带 Web UI 和 CLI
○支持 M1/M2
○对 macOS 支持好（包括 Apple Silicon），
○轻量级设计，适合资源受限设备（如树莓派、MacBook）。
●劣势：
○性能较弱，仅适合小型模型，（Mistral, LLaMA 2 7B 以下），
○缺乏分布式支持，大规模部署能力有限，不适合生产级部署，不支持大规模并发
●适用场景：
○适合个人开发者和轻量级应用
3.3 LMDeploy（阿里开源）
●简介：LMdeploy 是一个专为中文 LLM 优化的部署框架，支持多种量化策略和推理加速技术，特别适合中文场景的模型部署。
●核心技术：
由上海人工智能实验室开发，提供完整的模型量化、加速和部署工具链，支持多种硬件平台，特别适合资源受限场景
○Persistent Batch：连续批处理技术，吞吐量是 vLLM 的 1.8 倍。
○W4A16 量化：权重量化为 INT4，激活保持 FP16，推理速度是 FP16 的 2.4 倍，显存占用降低 60%。
○多设备支持：混合 DP+TP 模式，适配昇腾、海光等国产芯片。
○项目地址：https://github.com/InternLM/LMDeploy
●优势：
○支持 TensorRT / ONNX 推理后端，适合 NVIDIA 加速
○支持量化推理（INT8、AWQ）且量化效果优异，支持 KV Cache 量化，适合低显存环境。
○一键部署 Hugging Face 模型，兼容 OpenAI API。
○可和 InternLM、Qwen 等模型深度集成，推理优化深度好
●劣势：
○社区支持相对弱
○动态推理性能提升有限。
○对模型格式要求较高，主要适配 InternLM 生态，部分模型需手动转换
○文档不如 HF 完善，略复杂
●适用场景：
○适合企业级部署和边缘计算
3.3 TGI（Transformers-Serve，HuggingFace官方）
●简介：项目地址：https://github.com/huggingface/text-generation-inference
●优势：
○HuggingFace 官方支持：集成量化模型（支持量化（GPTQ、AWQ））、模型 Hub、AutoModel 等
○支持 Transformers 全生态
○REST/GRPC 支持，支持 token streaming
●劣势：
○内存较大
○性能不如 vLLM：吞吐略低于 vLLM
○API 兼容性差一些
○高性能部署需要手动调参
●适用场景：
○标准 Transformers 模型开箱即用，与HuggingFace Hub紧密集成的模型部署
○标准 API 部署、多语言推理
3.4 总结
需要极致推理性能，企业级的高并发服务部署推理——vllm
需要与 Transformers 无缝结合，如果已有 HuggingFace pipeline，TGI 直接集成——TGI
想快速在本地体验 LLaMA，快速迭代与本地开发Ollama 是零配置即用的首选。需要本地开发/演示——Ollama
企业 GPU 加速部署，如果用的是阿里系模型（如 Qwen）；边缘 / 低显存场景——LMDeploy最好 or Triton
对话模型仿 GPT-4 API——FastChat
6.19~6.27用多种框架部署大模型，分析吞吐率，延迟，产出报告
4.1 基础概念与定义
吞吐量高表示模型效率高，适合高并发、高负载环境部署，是衡量大模型推理系统性能的重要指标之一。延迟低说明响应快。
●吞吐量（↑）：模型每秒能生成多少个 token（批处理维度），举例如1000 tokens/s；
●Latency延迟（↓）：单个请求从输入到输出所需时间（单条请求），举例如0.3 秒生成完成一条回复；
4.2 部署推理报告
1.实验环境
●GPU：8×NVIDIA RTX 4090 24GB
●操作系统：Ubuntu 20.04.6 LTS
●CUDA版本：11.8
●python版本：3.10.18
2.vllm
创建新的虚拟环境用于推理部署：
conda create -n evaluate python=3.10 -y
conda activate evaluate

根据我的cuda版本和python版本，为了避免依赖冲突，我在https://github.com/vllm-project/vllm/tags网站中选择与我软件环境适配的whl包：vllm-0.6.1.post1+cu118-cp310-cp310-manylinux1_x86_64.whl进行安装:
pip install ./vllm-0.6.1.post1+cu118-cp310-cp310-manylinux1_x86_64.whl

然后创建 vllm 部署脚本vllm_deploy.py：
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

运行vllm部署脚本：
python vllm_deploy.py

最后运行结果截图如下：
中文prompt测试：
# 示例提示(中文)
prompts = [
    "介绍一下量子计算的基本原理",
    "简述人工智能的发展历程",
    "解释一下区块链技术",
    "什么是自然语言处理？",
    "请介绍机器学习中的监督学习和无监督学习"
]


各指标结果：模型加载耗时: 25.45秒
模型加载后显存占用: 17792 MB
平均延迟latency: 6.06秒/请求
平均吞吐率: 134.52 tokens/秒
VRAM使用变化: 9.09 MB
GPU显存使用变化: 40 MB
峰值GPU显存占用（模型加载）: 17792 MB

英文prompt测试：
# 示例提示(英文)
prompts = [
    "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
    "John writes 20 pages a day.  How long will it take him to write 3 books that are 400 pages each?",
    "Ed has 2 dogs, 3 cats and twice as many fish as cats and dogs combined. How many pets does Ed have in total?",
    "James buys 5 packs of beef that are 4 pounds each.  The price of beef is $5.50 per pound.  How much did he pay?？",
    "A store sells 20 packets of 100 grams of sugar every week. How many kilograms of sugar does it sell every week?"
]


各指标结果：模型加载耗时: 25.33秒（同上）
模型加载后显存占用: 17792 MB（同上）
平均延迟latency: 1.13秒/请求
平均吞吐率: 182.22 tokens/秒
VRAM使用变化: 18.55 MB
GPU显存使用变化: 52 MB
峰值GPU显存占用（模型加载）: 17792 MB

1.ollama
首先，官网下载linux的ollama：

curl -fsSL https://ollama.com/install.sh | sh

查看ollama是否安装成功：
# 命令行输入以下内容查看
ollama --version

显示以下内容提示安装成功：

由于ollama是自动安装在/usr/lib/ollama目录下的，如果非sudo用户，或者想要安装在指定目录，可以按照以下步骤：
# 下载 Ollama 安装包（不使用 sudo）
curl -L "https://ollama.com/download/ollama-linux-${ARCH}.tgz" -o ollama.tgz

# 创建解压目录
mkdir -p ollama

# 解压
tar -xzf ollama.tgz -C ./ollama
rm ollama.tgz

# 配置环境变量
export PATH="/tmp/deepseek-r1-distill-qwen2.5-7b/ollama/bin:$PATH"source ~/.bashrc

同样的测试有无安装成功；显示：

这是正常的，这表示成功运行了客户端程序 ollama，但它尝试连接 Ollama 的服务端（daemon）时，没有找到正在运行的 Ollama 后台服务进程，所以给出警告。因此要如下运行ollama后台服务程序（设置为后台运行）：
ollama serve &


此时再查看ollama版本，即显示：（正确）

由于ollama内部使用的是 llama.cpp 推理引擎，只能加载 GGUF 格式模型，因此
首先在modelscope下载gguf格式的Qwen2.5-7B-Instruct：（半精度float16，其余的量化文件一样的下载方法）
modelscope download --model second-state/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-f16.gguf --local_dir .

在进行模型部署前，需要Modelfile 构建自定义模型镜像：
# This is NOT a YAML file — 它是 Modelfile，语法类似 Dockerfile
FROM ./Qwen2.5-7B-Instruct-f16.gguf

# 推理参数（可按需调节）
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"

# 系统提示词（即 system prompt）默认值
SYSTEM "You are a helpful assistant."

# Prompt 模板，遵循 Qwen2.5-Instruct 的格式：
# 多轮对话通过 <|im_start|> 标签组织
TEMPLATE "<|im_start|>system\n{{ .System }}<|im_end|>\n<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n"

然后，构建 Ollama 模型镜像：
ollama create qwen2.5-7b-instruct -f Modelfile

构建成功提示：

然后就可以运行并测试模型：
ollama run qwen2.5-7b-instruct

运行成功截图：

多轮对话chat截图：

测试运行成功后，创建 Ollama 性能测试脚本ollama_deploy.py，对ollama的推理加速性能进行测试：
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


运行上述脚本：
python ollama_deploy.py

以下是性能测试结果截图：

各性能指标结果：
平均延迟: 4.51秒/请求
平均吞吐率: 28.79 tokens/秒
RAM加载增加: 14.05 MB
RAM推理增加: 0.00 MB
GPU加载增加: 0.00 MB
GPU推理增加: 0.00 MB
峰值GPU显存（加载期间）: 23124.19 MB
峰值GPU显存（推理期间）: 23109.62 MB

2.LMdeploy
首先安装LMdeploy：
pip install deploy

首先转换模型格式，把模型转换为 Turbomind 格式：
（原因：这里就很明显的感受到lmdeploy的缺陷了：lmdeploy（尤其是 lmdeploy.lite 和 serve 模块）是基于 Turbomind 后端的高性能推理框架，它不支持直接加载原始 HuggingFace 格式模型（pytorch_model.bin 或 safetensors），因此，在使用 lmdeploy 推理前，必须先把 HuggingFace 格式转换为 Turbomind 专属格式，即 .bin 和 .meta 文件的组合结构。）
lmdeploy convert qwen ./Models/qwen/Qwen2.5-7B-Instruct \
  --model-format hf \
  --dst-path ./Models/qwen/Qwen2.5-7B-Instruct-lmdeploy \
  --dtype float16 \
  --chat-template qwen

转换后的Qwen2.5-7B-Instruct-lmdeploy目录结构如下：

然后创建 LMdeploy 性能测试脚本lmdeploy_deploy.py：
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


运行上述脚本：
python LMdeploy_deploy.py

以下是性能测试结果截图：

各性能指标结果：
模型加载耗时: 1.64秒
模型加载后显存占用: 23897 MB
平均延迟latency: 33.66 秒/请求
平均吞吐率: 60.88 tokens/秒
加载阶段RAM增加: 556.69 MB
推理阶段RAM增加: 0.65 MB
加载阶段GPU显存增加: [23897, 0, 0, 0] MB
推理阶段GPU显存变化: [0, 0, 0, 0] MB
峰值GPU显存使用: [24265, 367, 367, 367] MB

6.25~6.28实际业务场景coa、cot数据微调qwen14b实践
●环境配置
首先新建一个conda虚拟环境swift-sft：
conda create -n swift-sft python=3.10

●模型下载
从modelscope下载模型：



●源码安装ms-swift
●微调数据准备
CoA数据1510条，数据格式如下：
{"conversations": 
    [
        {
            "from": "human", 
            "value": "2025年Q3的经营计划，欧洲市场的新订单增长和锂价的上升趋势是关键点，现金流问题暂不考虑。"
        }, 
        {
            "from": "gpt", 
            "value": "我将为您制定2025年Q3的经营计划，重点在于平衡订单增长与成本压力。首先调用<action1>销售计划智能体</action1>评估来自欧洲市场的新订单增长趋势，随后调用<action1.1>成本费用智能体</action1.1>分析锂价上升对成本的影响，接着调用<action1.1.1>盈利测算智能体</action1.1.1>评估这些因素对整体盈利的影响，最后调用<action1.1.1.1>资金管理智能体</action1.1.1.1>制定相应的经营策略。任务完成。"
        }
    ]
}

CoT数据1510条，数据格式如下：
{
    "conversations": 
        [
            {
                "from": "human", 
                "value": "在2025年第三季度的经营计划中，如何平衡欧洲市场的新订单增长和供应商缩短付款周期的需求？"
            }, 
            {
                "from": "gpt", 
                "value": "1.需要制定2025年第三季度的经营计划，任务流程包括销售计划、供应链管理和资金流优化三部分。-->2.销售计划：基于欧洲市场新订单快速增长的趋势，预测销售量并制定相应的市场策略。-->3.供应链管理：针对供应商缩短付款周期的要求，重新评估供应商关系，确保供应链稳定。-->4.资金流优化：虽然不考虑利润，但需调整内部资金流转策略，以适应缩短的付款周期，保证公司运营不受影响。任务完成。"
            }
        ]
}

而标准的用于监督微调的数据格式如下，其中system部分是可选的：
{
    "messages": 
        [
            {
                "role": "system", 
                "content": "<system>"
            }, 
            {
                "role": "user", 
                "content": "<query>"
            }, 
            {
                "role": "assistant", 
                "content": "<response>"
            }
        ]
}


●微调脚本编写
●微调训练
设计微调脚本：


Merge LoRA：
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --merge_lora true

使用CLI对LoRA训练的checkpoint进行推理：


参考
●https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b
●https://github.com/hiyouga/LLaMA-Factory
●https://llm-stats.com/
●https://mp.weixin.qq.com/s/bm7VLe__zxp0efVyFnzi4A
●https://aclanthology.org/2024.ccl-1.71.pdf
●https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
●https://ollama.com/
●...
市面30B~80B大模型效果对比
1.模型准备
**Qwen 2.5 (72B instruct)
**Qwen 2.5 (32B instruct)
**Llama-3.3 (70B instruct)
**Gemma 3 27B
Llama 3.1 70B Instruct
DeepSeek R1 Distill Qwen 32B

2.评测指标确定
https://docs.qq.com/sheet/DS1pvYUNFWnBzbHBj?tab=9w07ym
3.性能对比结果
https://docs.qq.com/sheet/DS1pvYUNFWnBzbHBj?tab=9w07ym
4.参考


