# 不可学习的注意力机制粗糙版实现
import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)
# 2nd input token is the query
query = inputs[1]
# 随机生成针对每个单词的注意力分数
attn_scores2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    # 让每一个其他单词向量和查询向量进行点积，得到query和目标单词向量的点积值
    attn_scores2[i] = torch.dot(x_i, query)
print(attn_scores2)
res = 0.
# 第一个单词和目标向量的点积具体实现
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print(res)
print(torch.dot(inputs[0], query))
# 把每一个点积的结果进行归一化，得到注意力分数
attn_weights_2_tmp = attn_scores2 / attn_scores2.sum()
print("Attention weights: ", attn_weights_2_tmp)
print("Sum: ", attn_weights_2_tmp.sum())


# 采用softmax函数对点积结果进行归一化，得到注意力分数
def softmax_naive(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=0)


attn_weights_2_naive = softmax_naive(attn_scores2)
print("Attention weights: ", attn_weights_2_naive)
print("Sum: ", attn_weights_2_naive.sum())

# 基础的softmax实现对于过大或过小的数据处理不太稳定，因此推荐采用torch的方式来实现
attn_weights_2 = torch.softmax(attn_scores2, dim=0)
print("Attention weights: ", attn_weights_2)
print("Sum: ", attn_weights_2.sum())

# 将注意力分数和输入向量相乘，得到经注意力计算后的加权query向量
query = inputs[1]
context_vec2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec2 += attn_weights_2[i] * x_i
print("context vec: ", context_vec2)

# 接下来计算对于每个单词分别作为query之后，注意力分数该如何计算
# step1：计算每个单词和其他单词点积的结果
attn_scores = torch.empty(6, 6)
# 手动实现
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
# 更快地采用矩阵乘法来实现
attn_scores = inputs @ inputs.T
print(attn_scores)
# step2：对每个单词和其他单词点积的结果进行归一化，得到注意力分数
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
# 验证每个单词的注意力分数之和为1
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))
# step3：将注意力分数和输入向量相乘，得到经注意力计算后的加权query向量
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# 注意力进阶：实现可学习的注意力机制
# 先来个简单的基础实现，针对一个单词的实现
x_2 = inputs[1]  # second input element
# embedding的维度可以和样本的维度相同，也可以和样本的维度不同
# 在GPT模型中，输入和输出embedding的维度通常是相同的
d_in = inputs.shape[1]  # the input embedding size, d=3
d_out = 2  # the output embedding size, d=2
torch.manual_seed(123)
# 为了方便演示，先将requires_grad设置为False
# 随机初始化参数矩阵
W_query = torch.nn.Parameter(torch.rand((d_in, d_out)), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand((d_in, d_out)), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand((d_in, d_out)), requires_grad=False)
# 然后针对第二个单词计算query、key和value
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)
# 可以把3维的词向量映射到2维空间
keys = inputs @ W_key
values = inputs @ W_value
print("keys shape: ", keys.shape)
print("values shape: ", values.shape)
key2 = keys[1]
attn_scores_2_2 = query_2.dot(key2)
print(attn_scores_2_2)
# 计算其他所有的单词和第二个单词的注意力关联系数
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)
# 把关联系数归一化
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
print(attn_weights_2)
# 计算单词2经向量化后的结果
context_vec_2_2 = attn_weights_2 @ values
print(context_vec_2_2)
import torch.nn as nn


# 基于nn.Parameter的可学习的注意力机制
class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = nn.Parameter(torch.rand((d_in, d_out)))
        self.W_k = nn.Parameter(torch.rand((d_in, d_out)))
        self.W_v = nn.Parameter(torch.rand((d_in, d_out)))

    def forward(self, x):
        keys = x @ self.W_k
        queries = x @ self.W_q
        values = x @ self.W_v
        attn_scores = queries @ keys.transpose(-1, -2)
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim=-1)
        context_vectors = attn_weights @ values
        return context_vectors


torch.manual_seed(123)
attn_v1 = SelfAttention_v1(d_in=3, d_out=2)
print(attn_v1(inputs))


# 基于nn.Linear的可学习的注意力机制
# 当Linear层没有bias时，相当于矩阵乘法
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)
        attn_scores = queries @ keys.transpose(-1, -2)
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim=-1)
        context_vectors = attn_weights @ values
        return context_vectors


torch.manual_seed(123)
attn_v2 = SelfAttention_v2(d_in=3, d_out=2)
print(attn_v2(inputs))
# 由于因果关系，所以在实际预测中，也就是一般的decoder中，需要掩盖掉未来的信息
# Reuse the query and key weight matrices of the
# SelfAttention_v2 object from the previous section for convenience
queries = attn_v2.W_q(inputs)
keys = attn_v2.W_k(inputs)
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

# 最简单的方式是创建一个下三角mask矩阵，然后和注意力分数相乘，这样未来的注意力就是0了
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
mask_simple = attn_weights * mask_simple
print(mask_simple)
# 为了保证处理后的概率和还是1，因此需要再进行归一化
row_sums = mask_simple.sum(dim=-1, keepdim=True)
mask_simple_norm = mask_simple / row_sums
print(mask_simple_norm)

# 对于softmax函数而言，当输入数据特别小时，softmax后的值会特别接近0，因此可以通过设置以恶搞很小的值来进行mask
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(mask)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

# 此外，还可以在注意力中加入dropout，随机mask一部分内容，来防止过拟合
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)  # dropout rate of 50%
example = torch.ones(6, 6)  # create a matrix of ones
print(dropout(example))
torch.manual_seed(123)
print(dropout(attn_weights))


# 然后，就来实现一个带dropout的完整注意力机制
class CasualAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)  # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)  # New

        context_vec = attn_weights @ values
        return context_vec


batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)  # 2 inputs with 6 tokens each, and each token has embedding dimension 3
torch.manual_seed(123)

context_length = batch.shape[1]
ca = CasualAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# 最后，就是多头注意力机制，下面是直接把各个头的结果拼接起来
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CasualAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1]  # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


# 更进阶的是增加一个权重矩阵，把各个头的结果加权重并整合成最后的注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
# (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
# 观察不同的头的计算详情
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

print(a @ a.transpose(2, 3))
first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_res)

second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)