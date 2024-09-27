# 采用tiktoken用于词元编码
import tiktoken
tik_tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, world. Is this-- a test?"
integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tik_tokenizer.decode(integers)
print(strings)
print(tik_tokenizer.n_vocab)
# 采用gpt-2进行词元编码
from bpe_openai_gpt2 import get_encoder, download_vocab
download_vocab()
orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")
integers = orig_tokenizer.encode(text)

print(integers)
strings = orig_tokenizer.decode(integers)

print(strings)
# 采用huggingface里的transformer进行词元编码
from transformers import GPT2Tokenizer

hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(hf_tokenizer(strings)["input_ids"])

