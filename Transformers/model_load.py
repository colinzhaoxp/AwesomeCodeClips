import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":

    # model 加载方式
    model_name_or_path = "name/path"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                torch_dtype=torch.bfloat16,
                                                low_cpu_mem_usage=True, # 避免超过内存
                                                attn_implementation="sdpa", # 注意力实现方式
                                                )
    
    # tokenizer 加载方式
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)