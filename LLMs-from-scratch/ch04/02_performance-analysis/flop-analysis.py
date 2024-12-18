"""
模型效率评估：FLOPs (Floating Point Operations Per Second) 通过计算模型在特定硬件上的浮点运算次数来评估模型计算开销
"""
# pip install -r requirements-extra.txt

from importlib.metadata import version
import importlib
import torch

print("thop version:", version('thop'))
print("torch version:", version('torch'))

from thop import profile
from previous_chapters import GPTModel

BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": True  # Query-key-value bias
}

model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = torch.randint(0, 50257, (2, 1024)).to(device)

for size in model_configs:
    BASE_CONFIG.update(model_configs[size])

    model = GPTModel(BASE_CONFIG).bfloat16()
    model.to(device)

    # MACS = multiply-accumulate operations
    # MACS are typically counted as two FLOPS (one multiply and one accumulate)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = 2 * macs
    # 模型大小 = 参数量 * 4 bytes / (1024 * 1024) MB
    # flops = macs * 2 (plus + multiply)
    print(f"{size:18}: {flops:.1e} FLOPS, {params} params")

    del model
    torch.cuda.empty_cache()