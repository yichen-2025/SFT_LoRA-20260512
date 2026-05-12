import logging
import os
from datetime import datetime
import pandas as pd

# 使用镜像网站，不然会很慢
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 禁用符号链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(),logging.FileHandler("system.log",encoding="utf-8")]
    )

# 导入transformers前需要设置环境变量↑
# 导入transformers库需要一段时间
logging.info("开始导入第三方库")
begin=datetime.now()

import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

end=datetime.now()
logging.info(f"导入第三方库耗时: {end-begin}")


class TestSystem:

    LABEL_NAME="output"     # 标签的名字

    # 系统提示词
    SYSTEM_PROMPT="""
    你是一个专业的网络安全专家，擅长识别DDoS等恶意流量。请根据以下网络流量特征，判断该流量是正常流量还是恶意流量。
    判断标准：
    恶意流量特征：
    - 平均包大小小于10字节或大于1000字节
    - 包长度标准差大于1000
    - 后向包比例大于0.6或小于0.1
    - 流间隔时间标准差大于5000000

    正常流量特征：
    - 平均包大小在10字节到1000字节之间
    - 包长度标准差在0到1000之间
    - 后向包比例在0.1到0.6之间
    - 流间隔时间标准差在0到5000000之间

    示例：
    1.平均包大小为1163.3字节。包长度均值为1057.55字节。包长度标准差为1853.44字节。平均前向段大小为8.67字节。平均后向段大小为1658.14字节。后向包长度均值为1658.14字节。后向包长度标准差为2137.3字节。后向包比例为0.0。SYN包比例为0.0。流间隔时间标准差为430865.8秒。回答：恶意流量
    2.平均包大小为34字节。包长度均值为22.67字节。包长度标准差为14.43字节。平均前向段大小为18.5字节。平均后向段大小为0.0字节。后向包长度均值为0.0字节。后向包长度标准差为0.0字节。后向包比例为0.0。SYN包比例为0.5。流间隔时间标准差为0秒。回答：正常流量
    只回答'正常流量'或'恶意流量'，严禁回答除此之外的其他内容，例如“恶性流量”，严禁添加任何解释。
    """

    # 把模型回复的文字映射为0或1
    @staticmethod
    def response_map(x):
        if "正常" in x:
            return 0
        elif "恶" in x:
            return 1
        else:
            return -1
        
    
    def __init__(
            self,
            model_file_path,
            test_file_name_or_path,
            lora_adapter_name_or_path=None,
            system_prompt=SYSTEM_PROMPT,
            label_name=LABEL_NAME
        ):
        """
        params:
            model_file_path: 模型文件路径
            test_file_name_or_path: 测试数据集文件名或路径
            system_prompt: 系统提示词
            label_name: 标签的名字
        """
        self.system_prompt=system_prompt
        self.label_name=label_name
        self.dataset=pd.read_csv(test_file_name_or_path)

        self.result={
            'label':[],
            'response':[],
            'model_response':[],
            'time':[]
        }

        for label in self.dataset[self.label_name]:
            self.result['label'].append(self.response_map(label))
        
        logging.info(f"开始加载模型: {model_file_path.split('/')[-1]}")

        self.tokenizer=AutoTokenizer.from_pretrained(model_file_path)
        self.model=AutoModelForCausalLM.from_pretrained(model_file_path,num_labels=2)
        self.lora_adapter_name=None
        if lora_adapter_name_or_path:
            self.lora_adapter_name=lora_adapter_name_or_path.split('/')[-1]
            self.model=PeftModel.from_pretrained(self.model, lora_adapter_name_or_path)
        self.model_name=model_file_path.split('/')[-1]
        
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=self.model.to(self.device)   # 将模型移动到指定设备

        logging.info(f"模型加载完成: {self.model_name}")
        logging.info(f"模型设备: {self.device}")
        if not torch.cuda.is_available():
            res=input("当前环境不支持GPU，是否继续？(y/n)")
            if res.lower()!="y":
                logging.info("用户取消预测")
                exit(0)
            logging.info("用户确认预测，继续执行")
    
    @staticmethod
    def response_processing(response):
        """
        对模型回复进行清理，提取出最后的回复
        params:
            response: 模型回复
        return:
            清理后的模型回复
        """
        if "assistant" in response:
            response=response.split("assistant")[-1].strip()
        
        response=response.replace("<|im_end|>","").replace("<|im_start|>","").strip()

        if "<think>" in response:
            # 提取 </think> 之后的内容
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            else:
                # 如果没有闭合标签，移除 <think> 及其后的内容直到找到答案
                response = response.split("<think>")[-1].strip()

        return response


    def send_prompt(self,prompt):
        """
        发送提示词到模型并获取回复
        params:
            prompt: 输入的提示词
        return:
            模型回复
        """
        
        # 把提示词发送给大模型
        messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        inputs=inputs.to(self.device)

        with torch.no_grad():   # 禁用梯度计算，节省内存和计算资源
            outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,

        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 对模型的回复进行清理，提取出最后的回复
        response=self.response_processing(response)

        logging.info(f"模型清理后的回复: {response}")
        
        return response

    def predict(self):
        """
        预测测试数据集中的每个样本
        return:
            预测结果
        """
        self.result['response']=[]
        self.result['model_response']=[]
        self.result['time']=[]

        for i,prompt in enumerate(self.dataset['input']):
            logging.info(f"开始处理第{i}条数据")
            begin=datetime.now()
            model_response=self.send_prompt(prompt)
            self.result['model_response'].append(model_response)
            self.result['response'].append(self.response_map(model_response))
            end=datetime.now()
            logging.info(f"第{i}条数据处理耗时: {end-begin}")
            self.result['time'].append((end-begin).total_seconds())

        logging.info("所有数据处理完成")
        return pd.DataFrame(self.result)

    def get_metrics_report(self):
        """
        获取模型在测试数据集上的指标报告
        return:
            指标报告
        """
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
        cm=confusion_matrix(self.result['label'],self.result['response'])

        metrics_report=pd.Series(
            index=[
                "model_name",
                "lora_adapter_name",
                "test_size",
                "label_0_count",
                "label_1_count",
                "response_0_count",
                "response_1_count",
                "response_error_count",
                "TP",
                "TN",
                "FP",
                "FN",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "total_time",
                "avg_time",
            ],
            data=[
                self.model_name,
                self.lora_adapter_name,
                len(self.dataset),
                self.result['label'].count(0),
                self.result['label'].count(1),
                self.result['response'].count(0),
                self.result['response'].count(1),
                self.result['response'].count(-1),
                cm[0,0],    # TP
                cm[1,1],    # TN
                cm[0,1],    # FP
                cm[1,0],    # FN
                accuracy_score(self.result['label'],self.result['response']),
                precision_score(self.result['label'],self.result['response']),
                recall_score(self.result['label'],self.result['response']),
                f1_score(self.result['label'],self.result['response']),
                sum(self.result['time']),
                sum(self.result['time'])/len(self.result['time']),
            ]
        )
        return metrics_report


    def execute(self):
        """
        执行测试
        return:
            预测结果
        """
        results=self.predict()

        result_file_path=f"results/result {datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        logging.info(f"预测结果已保存至: {result_file_path}")
        results.to_csv(result_file_path,index=False)

        report_file_path=f"test_reports/metrics_report {datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        metrics_report=self.get_metrics_report()
        metrics_report.to_csv(report_file_path,encoding="utf-8")
        logging.info(f"指标报告已保存至: {report_file_path}")
        
        return results,metrics_report
