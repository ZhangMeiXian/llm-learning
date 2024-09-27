# embedding和linear的区别
import torch
# Suppose we have the following 3 training examples,
# which may represent token IDs in a LLM context
idx = torch.tensor([2, 3, 1])
# The number of rows in the embedding matrix can be determined
# by obtaining the largest token ID + 1.
# If the highest token ID is 3, then we want 4 rows, for the possible
# token IDs 0, 1, 2, 3
num_idx = max(idx)+1

# The desired embedding dimension is a hyperparameter
out_dim = 5
# We use the random seed for reproducibility since
# weights in the embedding layer are initialized with
# small random values
torch.manual_seed(123)

embedding = torch.nn.Embedding(num_idx, out_dim)
print(embedding.weight)
print(embedding(torch.tensor([1])))
print(embedding(idx))
# embedding层是权重矩阵和输入向量直接相乘

onehot = torch.nn.functional.one_hot(idx)
print(onehot)
torch.manual_seed(123)
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
print(linear.weight)
linear.weight = torch.nn.Parameter(embedding.weight.T.detach())
print(linear(onehot.float()))
print(embedding(idx))
# Linear层是把输入向量转换成one-hot编码后的矩阵后，再和权重矩阵相乘