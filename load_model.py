import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(),logging.FileHandler("load_model.log",encoding="utf-8")]
    )

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_model(model_name,save_path=None):
    logging.info("开始导入第三方库")

    from transformers import AutoTokenizer,AutoModelForCausalLM

    logging.info("第三方库导入完成")

    # 加载分词器（Tokenizer）
    logging.info("开始加载tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info("tokenizer加载完成")

    # 加载模型（Model）
    logging.info("开始加载model")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    logging.info("model加载完成")

    if save_path is None:
        save_path = f"Qwen/{model_name.split('/')[-1]}"

    # 保存分词器（Tokenizer）
    logging.info("保存tokenizer")
    tokenizer.save_pretrained(save_path)


    # 保存模型（Model）
    logging.info("保存model")
    model.save_pretrained(save_path)

    logging.info(f"模型已保存到{save_path}")
