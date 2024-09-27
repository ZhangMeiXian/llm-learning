import re
# pip3 install tiktoken
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

with open("ch02/01_main-chapter-code/the-verdict.txt", encoding="utf8") as f:
    raw_text = f.read()

print("Total number of character: {}".format(len(raw_text)))
print(raw_text[:99])

# 1. 保留常用的标点符号、去除空格的分词处理
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print("Total number of preprocessed character: {}".format(len(preprocessed)))

# 2. 将单词处理成索引表示
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("Vocabulary size: {}".format(vocab_size))
vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(all_words):
    print((item, i))
    if i > 5: break


# 实现一个简单的文本向量化表示：文本到索引，索引到文本的映射
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[token] for token in preprocessed]

    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
print(tokenizer.decode(tokenizer.encode(text)))


# 增加一些特殊词汇的表示，比如：
# [BOS]：表示文本开始，beginning of sequence
# [EOS]: 表示文本结束，end of sequence
# [PAD]: 表示补全，padding
# [UNK]: 表示未知，unknown
# [MASK]: 表示遮蔽，mask
# <|endoftext|> : 表示文本结束，end of text


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)
# 词元编码的实现：https://github.com/openai/gpt-2/blob/master/src/encoder.py
byte_tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = byte_tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)
strings = byte_tokenizer.decode(integers)
print(strings)

# 在预测任务中，都是会掩盖掉后续的文本，基于历史窗口的数据进行预测
enc_text = byte_tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
print(len(enc_text))
# 设置一个上文窗口长度
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(byte_tokenizer.decode(context), "---->", byte_tokenizer.decode([desired]))


# 为了实现高效的数据加载，采用滑动窗口的方式获取历史窗口数据+目标数据
class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
# 3. 向量化，当文本转换成id后，下一步就是进行向量化
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
# 3.1 Embedding层会有初始化的权重可以将索引转换成对应的向量
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))
# 3.1 还需要考虑位置编码
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
context_length = max_length
# 简单地采用embedding层进行位置编码
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)