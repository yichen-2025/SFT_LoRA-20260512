import pandas as pd
from datetime import datetime

INSTRUCTION="""你是一个专业的网络安全专家，擅长识别DDoS等恶意流量。请根据以下网络流量特征，判断该流量是正常流量还是恶意流量。
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
    只回答'正常流量'或'恶意流量'，严禁添加任何解释。
    """

def preprocess(file_name,save_file_name=None,label_name='Label',benign_label='BENIGN'):
    """
    把原始数据集转换成处理后的数据集
    params:
        file_name: 原始数据集文件名
        save_file_name: 保存的处理后的数据集文件名
        label_name: 标签列名
        benign_label: 正常流量标签
    return:
        处理后的数据集
    """
    try:
        from Prompt import feature_to_natural_language
    except ImportError:
        raise ImportError("你是不是没把Prompt.py和preprocessing.py放到同一目录下？")

    df=pd.read_csv(f"raw_datasets/{file_name}")

    # 数据预处理
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.columns=df.columns.str.strip()

    df['instruction']=INSTRUCTION
    df['input']=df.apply(feature_to_natural_language,axis=1)
    df['output']=df[label_name].apply(lambda x:'正常流量' if x==benign_label else '恶意流量')

    df=df[['instruction','input','output']]

    if save_file_name is None:
        save_file_name=f"dataset {datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(f"processed/{save_file_name}",index=False)

    return df


def split_train_test(file_or_file_name,test_size=0.2,random_state=42):
    """
    划分训练集和测试集
    params:
        file_or_file_name: 处理后的数据集文件名或数据集
        test_size: 测试集比例
        random_state: 随机种子
    return:
        训练集和测试集
    """
    from sklearn.model_selection import train_test_split

    if isinstance(file_or_file_name,str):
        df=pd.read_csv(f"processed/{file_or_file_name}")
    else:
        df=file_or_file_name
    # 按output平均划分
    train_df, test_df = train_test_split(df, test_size=test_size,stratify=df['output'],random_state=random_state)
    
    train_df.to_csv(f"train/train_{len(train_df)}.csv",index=False)
    test_df.to_csv(f"test/test_{len(test_df)}.csv",index=False)
    return train_df, test_df

def to_json(train_file_or_file_name,save_json_name):
    """
    把训练集转换成json文件，用于微调模型
    params:
        train_file_or_file_name: 训练集文件名或数据集
        save_json_name: 保存的json文件名
    """
    if isinstance(train_file_or_file_name,str):
        df=pd.read_csv(f"train/{train_file_or_file_name}")
    else:
        df=train_file_or_file_name
    df.to_json(f"train/{save_json_name}",orient='records')

def shrink(datasets,size_0,size_1,random_state=42):
    """
    从数据集中中随机采集正样例和负样例各size_0和size_1条数据
    params:
        datasets: 数据集
        size_0: 采集的正常流量样本数
        size_1: 采集的恶意流量样本数
        random_state: 随机种子
    return:
        采集的样本数据集
    """
    import pandas as pd
    label_0=datasets[datasets['output']=='正常流量'].sample(size_0,random_state=random_state)
    label_1=datasets[datasets['output']=='恶意流量'].sample(size_1,random_state=random_state)
    return pd.concat([label_0,label_1],axis=0).sample(frac=1,random_state=random_state).reset_index(drop=True)