from importlib.metadata import version

pkgs = [
    "torch",
]
for p in pkgs:
    print(f"{p} version: {version(p)}")


# 统计GPU、CPU显存占用
import gc
import time
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def start_memory_tracking():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        print("CUDA is not available")


def print_memory_usage():
    max_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
    print(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)  # some buffer time to allow memory to clear
    torch.cuda.reset_peak_memory_stats()
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"Maximum GPU memory allocated: {max_memory_allocated:.1f} GB")


from previous_chapters import GPTModel


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-xl (1558M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])


start_memory_tracking()


model = GPTModel(BASE_CONFIG)
device = torch.device("cuda")
model.to(device)

print_memory_usage()


# Test if the model works (no need to track memory here)
test_input = torch.tensor([[1, 2, 3]]).to(device)
model.eval()

with torch.no_grad():
    model(test_input)


# Training code would go here...

model.train()
torch.save(model.state_dict(), "model.pth")


del model, test_input
cleanup()

# Then load pretrained weights

start_memory_tracking()

model = GPTModel(BASE_CONFIG)
model.to(device)

model.load_state_dict(
    torch.load("model.pth", map_location=device, weights_only=True)
)
model.to(device)
model.eval()

print_memory_usage()


# Test if the model works (no need to track memory here)
test_input = torch.tensor([[1, 2, 3]]).to(device)
model.eval()

with torch.no_grad():
    model(test_input)

del model, test_input
cleanup()


start_memory_tracking()

model = GPTModel(BASE_CONFIG).to(device)

state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)

print_memory_usage()

# Sequentially copy weights to the model's parameters
with torch.no_grad():
    for name, param in model.named_parameters():
        if name in state_dict:
            param.copy_(state_dict[name].to(device))
        else:
            print(f"Warning: {name} not found in state_dict.")

print_memory_usage()


# Test if the model works (no need to track memory here)
test_input = torch.tensor([[1, 2, 3]]).to(device)
model.eval()

with torch.no_grad():
    model(test_input)

del model, test_input, state_dict, param
cleanup()


import os
import psutil
from threading import Thread


def memory_usage_in_gb(func, *arg, **kwargs):
    process = psutil.Process(os.getpid())

    # 统计运行前的显存占用baseline
    baseline_mem = process.memory_info().rss / (1024 ** 3)  # 转换成GB

    # 监控多线程运行显存
    mem_usage, done = [], False

    def monitor_memory():
        while not done:
            mem_usage.append(process.memory_info().rss / (1024 ** 3))
            time.sleep(0.1)

    t = Thread(target=monitor_memory)
    t.start()

    # 运行
    func(*arg, **kwargs)

    # 停止监控
    done = True
    t.join()

    peak_mem_usage_gb = max(mem_usage) - baseline_mem
    return peak_mem_usage_gb


# 串行显存统计
def load_sequentially():
    start_memory_tracking()

    model = GPTModel(BASE_CONFIG).to(device)

    start_dict = torch.load("model.pth", map_location="cpu", weights_only=True)

    print_memory_usage()

    # Sequentially copy weights to the model's parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name].to(device))
            else:
                print(f"Warning: {name} not found in state_dict.")

    print_memory_usage()


peak_memory_used = memory_usage_in_gb(load_sequentially)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")
"""
Maximum GPU memory allocated: 6.4 GB
Maximum GPU memory allocated: 6.7 GB
-> Maximum CPU memory allocated: 6.3 GB
"""


# 当机器CPU显存很小，但是GPU显存很大时，可以使用GPU显存来加速
def load_sequentially_with_meta():
    start_memory_tracking()

    with torch.device("meta"):
        model = GPTModel(BASE_CONFIG)

    model = model.to_empty(device=device)

    state_dict = torch.load("model.pth", map_location=device, weights_only=True)

    print_memory_usage()

    # Sequentially copy weights to the model's parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name])
            else:
                print(f"Warning: {name} not found in state_dict.")

    print_memory_usage()

peak_memory_used = memory_usage_in_gb(load_sequentially_with_meta)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")
"""
Maximum GPU memory allocated: 12.8 GB
Maximum GPU memory allocated: 12.8 GB
-> Maximum CPU memory allocated: 1.3 GB
"""


def baseline():
    start_memory_tracking()

    model = GPTModel(BASE_CONFIG)
    model.to(device)

    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval();

    print_memory_usage()


peak_memory_used = memory_usage_in_gb(baseline)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")
"""
Maximum GPU memory allocated: 12.8 GB
-> Maximum CPU memory allocated: 4.4 GB
"""


def best_practices():
  with torch.device("meta"):
      model = GPTModel(BASE_CONFIG)

  model.load_state_dict(
      torch.load("model.pth", map_location=device, weights_only=True, mmap=True),
      assign=True
  )

  print_memory_usage()

peak_memory_used = memory_usage_in_gb(best_practices)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")
"""
Maximum GPU memory allocated: 6.4 GB
-> Maximum CPU memory allocated: 5.9 GB
"""


model = GPTModel(BASE_CONFIG)
# Assume `model` is your trained model
state_dict = model.state_dict()

# Create a directory to store individual parameter files
os.makedirs("model_parameters", exist_ok=True)

# Save each parameter tensor separately
for name, param in state_dict.items():
    torch.save(param.cpu(), f"model_parameters/{name}.pt")

del model


def load_individual_weights():

    start_memory_tracking()

    with torch.device("meta"):
        model = GPTModel(BASE_CONFIG)

    model = model.to_empty(device=device)

    print_memory_usage()
    param_dir = "model_parameters"

    with torch.no_grad():
        for name, param in model.named_parameters():
            weight_path = os.path.join(param_dir, f"{name}.pt")
            if os.path.exists(weight_path):
                param_data = torch.load(weight_path, map_location="cpu", weights_only=True)
                param.copy_(param_data)
                del param_data  # Free memory
            else:
                print(f"Warning: {name} not found in {param_dir}.")

    print_memory_usage()


peak_memory_used = memory_usage_in_gb(load_individual_weights)
print(f"-> Maximum CPU memory allocated: {peak_memory_used:.1f} GB")
"""
Maximum GPU memory allocated: 6.4 GB
Maximum GPU memory allocated: 6.4 GB
-> Maximum CPU memory allocated: 0.3 GB
"""