# ModelFineTuner.py
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fine-tune.log", encoding="utf-8")
    ]
)

logging.info("开始导入第三方库")
begin=datetime.now()

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback

modules_loading_time=datetime.now()-begin
logging.info(f"第三方库加载完成，耗时{modules_loading_time.total_seconds()}秒")

"""
数据集dataset内容
instruction: 指令
示例：你是一个专业的网络安全专家，擅长识别DDoS等恶意流量。请根据以下网络流量特征，判断该流量是正常流量还是恶意流量。
……
只回答'正常流量'或'恶意流量'，严禁添加任何解释。

input: 用自然语言对网络流量特征的描述
示例：你是一个网络安全专家，你需要判断以下网络流量是否为恶意流量。网络流量的特征如下：
平均包大小为{average_packet_size}字节，
……
请根据以上特征判断网络流量是否为恶意流量。只要回答“正常流量”或“恶意流量”即可。

output: 输出预测结果
示例：正常流量/恶意流量
"""


class ModelFineTuner:
    """
    模型微调类，用于对大语言模型进行LoRA微调
    
    Args:
        tokenizer: 分词器
        model: 预训练模型
        dataset: 训练数据集 (支持DataFrame或列表)
        output_dir: 模型保存目录
        lora_config: LoRA配置参数
        training_args: 训练参数
    """
    
    LABEL0_NAME="正常流量"
    LABEL1_NAME="恶意流量"

    def __init__(
        self,
        tokenizer,
        model,
        dataset,
        model_name = None,
        eval_size: float = 0.2,
        lora_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model_name = model_name
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset_size=len(dataset)
        self.eval_size=eval_size
        self.label0_count=sum(dataset['output']==self.LABEL0_NAME)
        self.label1_count=sum(dataset['output']==self.LABEL1_NAME)
        
        self.train_dataset, self.eval_dataset = self.split_train_eval(dataset,eval_size=eval_size)
        self.output_dir = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 初始化LoRA配置
        self.default_lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM", 
            "target_modules": ["q_proj", "k_proj","v_proj","o_proj"],
        }
        if lora_config:
            self.default_lora_config.update(lora_config)
        self.lora_config = LoraConfig(**self.default_lora_config)
        
        # 初始化训练参数
        self.default_training_args = { 
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 3,
            "weight_decay": 0.005,
            "learning_rate": 5e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "fp16": True,
            "load_best_model_at_end": True,
            "greater_is_better": False,
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 100,
            "save_total_limit": 1,
            "remove_unused_columns": True,
            "report_to": "none",
            "dataloader_num_workers": 0,
        }
        if training_args:
            self.default_training_args.update(training_args)
        self.training_args = TrainingArguments(**self.default_training_args)
        
        self.early_stop=EarlyStoppingCallback(early_stopping_patience=3)

        self.trainer = None

        self.process_time=0
        self.train_time=0
    
    @staticmethod
    def split_train_eval(datasets,eval_size=0.2,random_state=42):
        """
        划分训练集和验证集
        params:
            datasets: 数据集
            eval_size: 验证集比例
            random_state: 随机种子
        return:
            训练集和验证集
        """
        from sklearn.model_selection import train_test_split
        train_df, eval_df = train_test_split(datasets, test_size=eval_size,stratify=datasets['output'],random_state=random_state)
        return train_df, eval_df

    def _process_dataset(self, max_length: int = 512, format_type: str = "chat"):
        """
        处理数据集，转换为模型训练格式
        
        Args:
            max_length: 最大序列长度
            format_type: 格式类型，可选 "chat" 或 "instruction"
        """
        def format_sample(example):
            """格式化单个样本"""
            # 获取字段值
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')
            
            # 构建用户输入
            if input_text and input_text.strip():
                user_content = f"{instruction}\n{input_text}"
            else:
                user_content = instruction
            
            # 根据格式类型构建文本
            if format_type == "chat":
                formatted_text = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            else:  # instruction 格式
                formatted_text = f"### 指令：{instruction}\n### 输入：{input_text}\n### 输出：{output}"
            
            return {"text": formatted_text}
        
        # 处理不同类型的数据集

        if hasattr(self.train_dataset, 'iloc'):  # DataFrame
            train_texts=[format_sample(row) for _,row in self.train_dataset.iterrows()]
        elif isinstance(self.train_dataset, list):  # 列表
            train_texts = [format_sample(item) for item in self.train_dataset]
        else:
            raise ValueError("数据集格式不支持，请提供 DataFrame 或列表")
        

        if hasattr(self.eval_dataset, 'iloc'):  # DataFrame
            eval_texts=[format_sample(row) for _,row in self.eval_dataset.iterrows()]
        elif isinstance(self.eval_dataset, list):  # 列表
            eval_texts = [format_sample(item) for item in self.eval_dataset]
        else:
            raise ValueError("数据集格式不支持，请提供 DataFrame 或列表")

        # 分词函数
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None
            )

            tokenized['labels'] = tokenized['input_ids'].copy()

            # for i in range(len(tokenized['labels'])):
            #     text = examples["text"][i]
            #     # 找到 assistant 开始的位置
            #     assistant_start = text.find("<|im_start|>assistant\n")
            #     if assistant_start != -1:
            #         # 计算在 token 中的对应位置
            #         prompt_tokens = self.tokenizer(examples['text'][:assistant_start], add_special_tokens=False)["input_ids"]
            #         # 将 prompt 部分的 label 盖住
            #         tokenized['labels'][i][:len(prompt_tokens)] = [-100] * len(prompt_tokens)

            # if isinstance(examples["text"], str):
            #     tokenized = self.tokenizer(
            #         examples["text"],
            #         truncation=True,
            #         padding="max_length",
            #         max_length=max_length,
            #         return_tensors=None
            #     )
                
            #     tokenized['labels'] = tokenized['input_ids'].copy()
                
            #     # 找到 assistant 开始的位置
            #     assistant_start = examples["text"].find("<|im_start|>assistant\n")
            #     if assistant_start != -1:
            #         # 计算在 token 中的对应位置
            #         prompt_tokens = self.tokenizer(examples["text"][:assistant_start], add_special_tokens=False)["input_ids"]
            #         # 将 prompt 部分的 label 盖住
            #         tokenized['labels'][:len(prompt_tokens)] = [-100] * len(prompt_tokens)
            # else:
            #     # 处理批次样本的情况
            #     tokenized = self.tokenizer(
            #         examples["text"],
            #         truncation=True,
            #         padding="max_length",
            #         max_length=max_length,
            #         return_tensors=None
            #     )
                
            #     tokenized['labels'] = tokenized['input_ids'].copy()
                
            #     for i in range(len(tokenized['labels'])):
            #         text = examples["text"][i]
            #         # 找到 assistant 开始的位置
            #         assistant_start = text.find("<|im_start|>assistant\n")
            #         if assistant_start != -1:
            #             # 计算在 token 中的对应位置
            #             prompt_tokens = self.tokenizer(text[:assistant_start], add_special_tokens=False)["input_ids"]
            #             # 确保 prompt_tokens 长度不超过 max_length
            #             prompt_len = min(len(prompt_tokens), max_length)
            #             # 将 prompt 部分的 label 盖住
            #             tokenized['labels'][i][:prompt_len] = [-100] * prompt_len

            return tokenized
        
        self.train_dataset=Dataset.from_list(train_texts)
        self.eval_dataset=Dataset.from_list(eval_texts)
        
        # 进行分词
        self.train_dataset = self.train_dataset.map(
            tokenize_function, 
            batched=True,
            batch_size=100, 
            remove_columns=["text"]
        )

        self.eval_dataset = self.eval_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=["text"]
        )
    
    def get_lora_report(self):
        import pandas as pd
        lora_report=pd.Series(
            index=[
                "model_name",
                "output_dir",
                "r",
                "lora_alpha",
                "lora_dropout",
                "num_train_epochs",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "lr_scheduler_type",
                "fp16",
                "learning_rate",
                "dataset_size",
                "eval_size",
                "label0_count",
                "label1_count",
                "process_time",
                "train_time"
            ],
            data=[
                self.model_name,
                self.output_dir,
                self.default_lora_config["r"],
                self.default_lora_config["lora_alpha"],
                self.default_lora_config["lora_dropout"],
                self.default_training_args["num_train_epochs"],
                self.default_training_args["per_device_train_batch_size"],
                self.default_training_args["gradient_accumulation_steps"],
                self.default_training_args["lr_scheduler_type"],
                self.default_training_args["fp16"],
                self.default_training_args["learning_rate"],
                self.dataset_size,
                self.eval_size,
                self.label0_count,
                self.label1_count,
                self.process_time,
                self.train_time
            ]
        )
        return lora_report

    def train(self, max_length: int = 512, format_type: str = "chat"):
        """
        训练模型
        
        Args:
            max_length: 最大序列长度
            format_type: 格式类型，可选 "chat" 或 "instruction"
        
        Returns:
            tokenizer: 分词器
            model: 微调后的模型
        """
        from transformers import DataCollatorForLanguageModeling

        logging.info("开始处理数据集...")
        begin=datetime.now()

        self._process_dataset(max_length, format_type)

        self.process_time=datetime.now()-begin
        logging.info(f"数据处理完成，耗时{self.process_time.total_seconds()}秒")
        
        logging.info("开始给模型添加LoRA适配器...")
        self.model = get_peft_model(self.model, self.lora_config)
        logging.info("模型添加LoRA适配器完成")
        
        self.model.gradient_checkpointing_enable()
        self.model.print_trainable_parameters()

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

        logging.info(f"当前设备：{self.device}")
        if not torch.cuda.is_available():
            logging.warning("GPU不可用，将使用CPU训练，这可能会非常慢")
            try:
                user_input = input("是否继续训练？（y/n）：")
                if user_input.lower() != "y":
                    logging.info("用户取消训练")
                    return self.tokenizer, self.model
            except KeyboardInterrupt:
                logging.info("用户中断操作")
                return self.tokenizer, self.model
        

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=[self.early_stop]
        )
        
        logging.info("开始训练模型...")
        begin = datetime.now()
        
        self.trainer.train()

        self.train_time=datetime.now()-begin
        logging.info(f"模型训练完成，耗时{self.train_time.total_seconds()}秒")
        
        # 保存微调后的模型
        self.output_dir = f"Qwen/fine-tuned-qwen-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logging.info(f"模型已保存至：{self.output_dir}")
        
        lora_report=self.get_lora_report()
        lora_report.to_csv(f"lora_reports/lora_report {datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        return self.tokenizer, self.model
    

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        dataset,
        eval_size: float = 0.2,
        lora_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None
    ):
        """
        从预训练模型创建微调器
        
        Args:
            model_name_or_path: 模型名称或路径
            dataset: 训练数据集
            
            lora_config: LoRA配置参数
            training_args: 训练参数
        
        Returns:
            ModelFineTuner: 微调器实例
        """
        model_name=model_name_or_path.split('/')[-1]

        logging.info(f"加载模型：{model_name_or_path}")
        begin = datetime.now()
        
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        logging.info(f"模型加载完成，耗时：{datetime.now() - begin}")
        
        return cls(
            tokenizer=tokenizer,
            model=model,
            dataset=dataset,
            model_name=model_name,
            eval_size=eval_size,
            lora_config=lora_config,
            training_args=training_args
        )


def shrink(datasets,size,random_state=42):
    """
    从数据集中中随机采集正样例和负样例各size条数据
    params:
        datasets: 数据集
        size: 采集的样本数
        random_state: 随机种子
    return:
        采集的样本数据集
    """
    import pandas as pd
    label_0=datasets[datasets['output']=='正常流量'].sample(size,random_state=random_state)
    label_1=datasets[datasets['output']=='恶意流量'].sample(size,random_state=random_state)
    return pd.concat([label_0,label_1],axis=0).sample(frac=1,random_state=random_state).reset_index(drop=True)


