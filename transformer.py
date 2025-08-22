# Transformer模型实现
# 这是Google在2017年论文"Attention Is All You Need"中提出的Transformer模型的完整实现
# Transformer是现代深度学习中最重要的模型架构之一，BERT、GPT等模型都基于它构建

import numpy as np
import torch.nn as nn
from datasets import *

# ===== 模型超参数配置 =====
# 这些参数控制着Transformer模型的大小和复杂度
d_model = 512   # 词嵌入的维度，也是模型内部所有向量的统一维度
d_ff = 2048     # 前馈神经网络中隐藏层的维度，通常是d_model的4倍
d_k = d_v = 64  # 注意力机制中Key和Value的维度，通常是d_model除以注意力头数
n_layers = 6    # Encoder和Decoder各自的层数，原论文中都是6层
n_heads = 8     # 多头注意力的头数，让模型能从不同角度关注信息


class PositionalEncoding(nn.Module):
    """
    位置编码类 - 为输入序列添加位置信息
    
    由于Transformer没有循环结构，无法感知词语的位置关系，
    因此需要通过位置编码来告诉模型每个词在句子中的位置。
    使用正弦和余弦函数生成位置编码，这样可以让模型学习到相对位置关系。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # 防止过拟合的dropout层
        
        # 创建位置编码表，大小为 [max_len, d_model]
        # 对于每个位置pos和每个维度i，计算位置编码值
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        
        # 对偶数维度使用sin函数，对奇数维度使用cos函数
        # 这样设计可以让模型更容易学习到位置之间的相对关系
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # 偶数维度用sin
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # 奇数维度用cos
        
        # 将位置编码表转换为PyTorch张量并移到GPU
        self.pos_table = torch.FloatTensor(pos_table).cuda()

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, seq_len, d_model]
        """
        前向传播：将位置编码加到词嵌入上
        
        Args:
            enc_inputs: 词嵌入向量 [batch_size, seq_len, d_model]
        Returns:
            加上位置编码后的向量 [batch_size, seq_len, d_model]
        """
        # 将对应长度的位置编码加到输入上
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs.cuda())


def get_attn_pad_mask(seq_q, seq_k):
    """
    生成填充掩码 - 防止注意力机制关注到填充位置
    
    在处理不同长度的句子时，我们会用0来填充较短的句子。
    这个函数生成掩码，告诉注意力机制哪些位置是填充的，不应该被关注。
    
    Args:
        seq_q: 查询序列 [batch_size, len_q]
        seq_k: 键序列 [batch_size, len_k]
    Returns:
        填充掩码 [batch_size, len_q, len_k]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    
    # 找出seq_k中值为0的位置（填充位置），标记为True
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k]
    
    # 将掩码扩展到查询序列的每个位置
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    """
    生成序列掩码 - 防止解码器看到未来的信息
    
    在训练解码器时，我们不希望当前位置能看到后面位置的信息，
    因为在实际推理时是没有这些信息的。这个函数生成上三角掩码来实现这个目标。
    
    Args:
        seq: 目标序列 [batch_size, tgt_len]
    Returns:
        序列掩码 [batch_size, tgt_len, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    
    # 生成上三角矩阵，对角线上方的元素为1，表示这些位置需要被掩码
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    
    # 转换为PyTorch张量
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力 - Transformer的核心注意力机制
    
    这是"Attention Is All You Need"论文中提出的注意力计算方法：
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
    通过计算查询(Q)和键(K)的相似度来决定对值(V)的关注程度。
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        前向传播：计算缩放点积注意力
        
        Args:
            Q: 查询矩阵 [batch_size, n_heads, len_q, d_k]
            K: 键矩阵 [batch_size, n_heads, len_k, d_k]
            V: 值矩阵 [batch_size, n_heads, len_v(=len_k), d_v]
            attn_mask: 注意力掩码 [batch_size, n_heads, seq_len, seq_len]
        Returns:
            context: 上下文向量 [batch_size, n_heads, len_q, d_v]
            attn: 注意力权重 [batch_size, n_heads, len_q, len_k]
        """
        # 步骤1: 计算注意力分数 = Q × K^T / √d_k
        # 除以√d_k是为了防止分数过大导致softmax梯度消失
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        
        # 步骤2: 应用掩码，将需要忽略的位置设为很小的负数
        # 这样经过softmax后这些位置的权重接近0
        scores.masked_fill_(attn_mask, -1e9)
        
        # 步骤3: 对注意力分数应用softmax，得到注意力权重
        attn = nn.Softmax(dim=-1)(scores)
        
        # 步骤4: 用注意力权重对值向量进行加权求和
        context = torch.matmul(attn, V)
        
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 - 让模型能从多个角度关注信息
    
    多头注意力的核心思想是：
    1. 将输入投影到多个不同的子空间（多个头）
    2. 在每个子空间中独立计算注意力
    3. 将所有头的结果拼接起来再做一次线性变换
    
    这样可以让模型同时关注不同类型的信息，比如语法关系、语义关系等。
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 定义Q、K、V的线性变换层
        # 每个头都有自己的Q、K、V，所以输出维度是 d_k * n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # 查询变换
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)  # 键变换
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)  # 值变换
        
        # 最后的线性变换层，将多头结果合并
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        多头注意力的前向传播
        
        Args:
            input_Q: 查询输入 [batch_size, len_q, d_model]
            input_K: 键输入 [batch_size, len_k, d_model]
            input_V: 值输入 [batch_size, len_v(=len_k), d_model]
            attn_mask: 注意力掩码 [batch_size, seq_len, seq_len]
        Returns:
            output: 多头注意力输出 [batch_size, len_q, d_model]
            attn: 注意力权重 [batch_size, n_heads, len_q, len_k]
        """
        # 保存残差连接的输入
        residual, batch_size = input_Q, input_Q.size(0)
        
        # 步骤1: 线性变换得到Q、K、V，然后重塑为多头形式
        # 将 [batch_size, seq_len, d_model] 变换为 [batch_size, n_heads, seq_len, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        
        # 步骤2: 将掩码扩展到所有注意力头
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        # 步骤3: 对每个头独立计算缩放点积注意力
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        
        # 步骤4: 将多头结果拼接起来
        # [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        
        # 步骤5: 最后的线性变换
        output = self.fc(context)  # [batch_size, len_q, d_model]
        
        # 步骤6: 残差连接 + 层归一化
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    """
    位置前馈网络 - Transformer中的前馈神经网络层
    
    这是一个简单的两层全连接网络：
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    作用是对每个位置的表示进行非线性变换，增强模型的表达能力。
    注意这个网络是逐位置应用的，即对序列中每个位置独立处理。
    """
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # 两层全连接网络：d_model -> d_ff -> d_model
        # 中间层维度通常是输入维度的4倍
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),    # 第一层：扩展维度
            nn.ReLU(),                               # ReLU激活函数
            nn.Linear(d_ff, d_model, bias=False))    # 第二层：恢复原始维度

    def forward(self, inputs):
        """
        前向传播：对输入进行位置前馈变换
        
        Args:
            inputs: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        residual = inputs  # 保存用于残差连接
        output = self.fc(inputs)  # 前馈网络变换
        
        # 残差连接 + 层归一化
        return nn.LayerNorm(d_model).cuda()(output + residual)


class EncoderLayer(nn.Module):
    """
    编码器层 - Transformer编码器的基本构建块
    
    每个编码器层包含两个主要组件：
    1. 多头自注意力机制：让每个位置关注序列中的所有位置
    2. 位置前馈网络：对每个位置进行非线性变换
    
    每个组件后面都有残差连接和层归一化。
    """
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()    # 多头自注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()       # 位置前馈网络

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        编码器层的前向传播
        
        Args:
            enc_inputs: 编码器输入 [batch_size, src_len, d_model]
            enc_self_attn_mask: 自注意力掩码 [batch_size, src_len, src_len]
        Returns:
            enc_outputs: 编码器输出 [batch_size, src_len, d_model]
            attn: 注意力权重 [batch_size, n_heads, src_len, src_len]
        """
        # 步骤1: 多头自注意力
        # 输入同时作为Q、K、V，实现自注意力
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        
        # 步骤2: 位置前馈网络
        enc_outputs = self.pos_ffn(enc_outputs)
        
        return enc_outputs, attn

class Encoder(nn.Module):
    """
    Transformer编码器 - 将输入序列编码为上下文表示
    
    编码器的主要功能是理解输入序列，生成每个位置的上下文表示。
    它由以下组件构成：
    1. 词嵌入层：将词汇转换为向量
    2. 位置编码：添加位置信息
    3. N个编码器层：逐层提取更高级的特征
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  # 源语言词嵌入
        self.pos_emb = PositionalEncoding(d_model)            # 位置编码
        # 堆叠多个编码器层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        编码器的前向传播
        
        Args:
            enc_inputs: 输入序列的词汇索引 [batch_size, src_len]
        Returns:
            enc_outputs: 编码后的表示 [batch_size, src_len, d_model]
            enc_self_attns: 各层的自注意力权重列表
        """
        # 步骤1: 词嵌入 - 将词汇索引转换为向量
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        
        # 步骤2: 添加位置编码
        enc_outputs = self.pos_emb(enc_outputs)  # [batch_size, src_len, d_model]
        
        # 步骤3: 生成填充掩码，防止关注填充位置
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        
        # 步骤4: 通过所有编码器层
        enc_self_attns = []  # 保存每层的注意力权重
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        
        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module):
    """
    解码器层 - Transformer解码器的基本构建块
    
    每个解码器层包含三个主要组件：
    1. 掩码多头自注意力：让当前位置只能关注之前的位置
    2. 编码器-解码器注意力：让解码器关注编码器的输出
    3. 位置前馈网络：对每个位置进行非线性变换
    
    每个组件后面都有残差连接和层归一化。
    """
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()   # 解码器自注意力
        self.dec_enc_attn = MultiHeadAttention()    # 编码器-解码器注意力
        self.pos_ffn = PoswiseFeedForwardNet()      # 位置前馈网络

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        解码器层的前向传播
        
        Args:
            dec_inputs: 解码器输入 [batch_size, tgt_len, d_model]
            enc_outputs: 编码器输出 [batch_size, src_len, d_model]
            dec_self_attn_mask: 解码器自注意力掩码 [batch_size, tgt_len, tgt_len]
            dec_enc_attn_mask: 编码器-解码器注意力掩码 [batch_size, tgt_len, src_len]
        Returns:
            dec_outputs: 解码器输出 [batch_size, tgt_len, d_model]
            dec_self_attn: 自注意力权重 [batch_size, n_heads, tgt_len, tgt_len]
            dec_enc_attn: 编码器-解码器注意力权重 [batch_size, n_heads, tgt_len, src_len]
        """
        # 步骤1: 掩码多头自注意力
        # 防止当前位置看到未来的信息
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        
        # 步骤2: 编码器-解码器注意力
        # 让解码器关注编码器的输出，实现源语言到目标语言的信息传递
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        
        # 步骤3: 位置前馈网络
        dec_outputs = self.pos_ffn(dec_outputs)
        
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    """
    Transformer解码器 - 根据编码器输出生成目标序列
    
    解码器的主要功能是根据编码器的输出和已生成的部分目标序列，
    逐步生成完整的目标序列。它由以下组件构成：
    1. 目标语言词嵌入层
    2. 位置编码
    3. N个解码器层
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)  # 目标语言词嵌入
        self.pos_emb = PositionalEncoding(d_model)            # 位置编码
        # 堆叠多个解码器层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        解码器的前向传播
        
        Args:
            dec_inputs: 目标序列的词汇索引 [batch_size, tgt_len]
            enc_inputs: 源序列的词汇索引 [batch_size, src_len]
            enc_outputs: 编码器输出 [batch_size, src_len, d_model]
        Returns:
            dec_outputs: 解码器输出 [batch_size, tgt_len, d_model]
            dec_self_attns: 各层自注意力权重列表
            dec_enc_attns: 各层编码器-解码器注意力权重列表
        """
        # 步骤1: 目标语言词嵌入
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        
        # 步骤2: 添加位置编码
        dec_outputs = self.pos_emb(dec_outputs).cuda()  # [batch_size, tgt_len, d_model]
        
        # 步骤3: 生成各种掩码
        # 填充掩码：防止关注填充位置
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()
        # 序列掩码：防止看到未来信息
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()
        # 组合掩码：同时应用填充掩码和序列掩码
        dec_self_attn_mask = torch.gt(
            (dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda()
        # 编码器-解码器注意力掩码：防止关注编码器的填充位置
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        
        # 步骤4: 通过所有解码器层
        dec_self_attns, dec_enc_attns = [], []  # 保存注意力权重
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    """
    完整的Transformer模型 - 序列到序列的转换模型
    
    这是"Attention Is All You Need"论文中提出的完整Transformer架构。
    主要用于机器翻译等序列到序列的任务，将源语言序列转换为目标语言序列。
    
    模型架构：
    1. 编码器：理解源语言序列，生成上下文表示
    2. 解码器：根据编码器输出和已生成的目标序列，生成下一个词
    3. 输出投影层：将解码器输出转换为词汇表上的概率分布
    """
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder().cuda()     # 编码器
        self.Decoder = Decoder().cuda()     # 解码器
        # 输出投影层：将d_model维度映射到目标词汇表大小
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs):
        """
        Transformer的前向传播
        
        Args:
            enc_inputs: 源序列的词汇索引 [batch_size, src_len]
            dec_inputs: 目标序列的词汇索引 [batch_size, tgt_len]
        Returns:
            dec_logits: 输出词汇的logits [batch_size * tgt_len, tgt_vocab_size]
            enc_self_attns: 编码器各层自注意力权重
            dec_self_attns: 解码器各层自注意力权重
            dec_enc_attns: 解码器各层编码器-解码器注意力权重
        """
        # 步骤1: 编码器处理源序列
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)
        
        # 步骤2: 解码器根据编码器输出和目标序列生成表示
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)
        
        # 步骤3: 输出投影，得到每个位置在词汇表上的分数
        dec_logits = self.projection(dec_outputs)  # [batch_size, tgt_len, tgt_vocab_size]
        
        # 重塑为二维张量，方便计算损失函数
        return (dec_logits.view(-1, dec_logits.size(-1)), 
                enc_self_attns, dec_self_attns, dec_enc_attns)
