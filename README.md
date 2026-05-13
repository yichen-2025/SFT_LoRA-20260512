# 文件说明
- `Prompt.py`用于把流量样本映射为文本描述
- `preprocessing.py`用于处理数据集
- `load_model.py`用于下载大模型
- `system.py`用于在测试集上测试大模型性能
- `ModelFineTuner.py`用于微调大模型（SFT+LoRA）
- `日志文件`：带log后缀的是日志文件，可删除

# 下载大模型
在`main.py`文件中运行：
```python
from load_model import load_model
load_model(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",   # 模型名称
    save_path="Qwen/qwen2.5-1.5b-instruct"     # 模型保存路径，可不填
    )
```
运行完成后，可在`Qwen`文件夹中查看下载的模型

`model_name`可填`Qwen2.5-1.5B-Instruct`或`Qwen3-1.7B`等。

# 处理数据集
以`raw_datasets`中的`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`为例，将数据集的数值特征映射为文本描述。

映射后的结果格式是这个样子：
```plaintext
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
```
在`main.py`文件中运行：
```python
import pandas as pd
from preprocessing import preprocess

preprocess(
    file_name="Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",   # 原始数据集文件名
    save_file_name="dataset_0.csv",   # 保存的处理后的数据集文件名,可不填
    label_name="Label",   # 标签列名,默认为'Label'
    benign_label="BENIGN"    # 正常流量标签,默认为'Benign'
    )
# 注意：每个参数都要写对，参考preprocessing.py文件
```

运行完成后，可在`processed`文件夹中查看处理后的数据集

要从这些海量数据中获取子集，在`main.py`文件中运行：
```python
import pandas as pd
from preprocessing import shrink

df = pd.read_csv("processed/dataset_0.csv") # 假设处理后的数据集文件名为dataset_0.csv
df = shrink(
    datasets=df,    # 输入数据集
    size_0=500,     # 正常流量样本数
    size_1=500,     # 恶意流量样本数
    random_state=42 # 随机种子，可不填
)
df.to_csv("test/test_1.csv", index=False) # 保存子集
```
用这个方法课获得小样本训练集/测试集，把生成的文件放在train或test文件夹中

# 测试模型
在`main.py`文件中运行：
```python
from system import TestSystem
system=TestSystem(
    model_file_path="Qwen/qwen2.5-1.5b-instruct",   # 模型路径
    test_file_name_or_path="test/test_1.csv",   # 测试集路径
    lora_adapter_name_or_path=None,   # LoRA适配器文件路径，由ModelFineTuner.py生成，可不填
)
system.execute()    # 执行
```
运行完成后，可在`test_reports`文件夹中查看测试结果,包括准确率、精确率、召回率、F1值、推理时间等。

# 微调模型
首先在数据集处理部分准备好相应的训练集。

在`main.py`文件中运行：
```python
from ModelFineTuner import ModelFineTuner
tuner=ModelFineTuner.from_pretrained(
    model_name_or_path="Qwen/qwen2.5-1.5b-instruct",   # 基座模型路径
    dataset="train/train_1.csv",   # 训练集路径
    eval_size=0.2,   # 验证集比例，默认0.2，可不填
    # LoRA配置和训练参数，可根据需要传入字典
    lora_config=None,
    training_args=None,
)
tuner.train()    # 开始训练
```
运行完成后，可在`Qwen`文件夹中找到对应的LoRA适配器文件，在`lora_reports`文件夹中查看微调结果。

# 可改进的地方
1.`ModelFineTuner.py`文件还有很大的提升空间。可以问问AI，可以怎么优化。

2.可以优化提示词，引入思维链。和提示词有关的文件有`preprocessing.py`、`system.py`、`Prompt.py`。

3.可以参照上述方法，用于其他的数据集。
