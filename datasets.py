# 数据集处理模块 - 用于Transformer模型的训练数据准备
# 参考资料: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
"""
这个模块负责处理Transformer模型的训练数据，主要功能包括：
1. 定义中英文对照的训练句子
2. 构建源语言和目标语言的词汇表
3. 将文本转换为模型可以处理的数字索引
4. 提供PyTorch数据集接口

这是一个简化的机器翻译数据集，用于演示Transformer模型的基本训练过程。
"""

import torch
import torch.utils.data as Data

# ==================== 训练数据定义 ====================
# 训练句子：[源语言(中文), 目标语言输入(英文), 目标语言输出(英文)]
# 每个句子包含三个部分：编码器输入、解码器输入、解码器期望输出
sentences = [
    ['我 是 学 生 P', 'S I am a student', 'I am a student E'],     # 句子1：我是学生
    ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'], # 句子2：我喜欢学习  
    ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']              # 句子3：我是男生
]
# 特殊符号说明：
# S: 开始符号(Start) - 告诉解码器开始生成序列
# E: 结束符号(End) - 告诉解码器序列生成结束
# P: 填充符号(Padding) - 用于补齐不同长度的句子

# ==================== 源语言词汇表(中文) ====================
# 源语言词汇表：将中文字符映射为数字索引
src_vocab = {
    'P': 0,   # 填充符号
    '我': 1,  # 我
    '是': 2,  # 是
    '学': 3,  # 学
    '生': 4,  # 生
    '喜': 5,  # 喜
    '欢': 6,  # 欢
    '习': 7,  # 习
    '男': 8   # 男
}

# 源语言反向词汇表：从索引映射回中文字符（用于结果展示）
src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)  # 源语言词汇表大小

# ==================== 目标语言词汇表(英文) ====================
# 目标语言词汇表：将英文单词映射为数字索引
tgt_vocab = {
    'P': 0,         # 填充符号
    'S': 1,         # 开始符号
    'E': 2,         # 结束符号
    'I': 3,         # I
    'am': 4,        # am
    'a': 5,         # a
    'student': 6,   # student
    'like': 7,      # like
    'learning': 8,  # learning
    'boy': 9        # boy
}

# 目标语言反向词汇表：从索引映射回英文单词（用于结果展示）
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}
tgt_vocab_size = len(tgt_vocab)  # 目标语言词汇表大小

# ==================== 序列长度设置 ====================
# 根据训练数据确定序列的最大长度
src_len = len(sentences[0][0].split(" "))  # 编码器输入的最大长度（中文句子长度）
tgt_len = len(sentences[0][1].split(" "))  # 解码器输入输出的最大长度（英文句子长度）


# ==================== 数据转换函数 ====================
def make_data():
    """
    将文本句子转换为模型可以处理的数字索引张量
    
    这个函数的主要作用：
    1. 遍历所有训练句子
    2. 将中文字符转换为源语言词汇表中的索引
    3. 将英文单词转换为目标语言词汇表中的索引
    4. 返回三个张量：编码器输入、解码器输入、解码器期望输出
    
    Returns:
        enc_inputs: 编码器输入张量 [num_samples, src_len]
        dec_inputs: 解码器输入张量 [num_samples, tgt_len] 
        dec_outputs: 解码器期望输出张量 [num_samples, tgt_len]
    """
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    
    # 遍历每个训练句子
    for i in range(len(sentences)):
        # 处理编码器输入（中文句子）
        # 将中文字符序列转换为对应的词汇表索引
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        
        # 处理解码器输入（英文句子，包含开始符号S）
        # 解码器输入用于告诉模型当前已经生成了哪些词
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        
        # 处理解码器期望输出（英文句子，包含结束符号E）
        # 这是模型应该预测的正确答案
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        
        # 将处理好的数据添加到列表中
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    
    # 转换为PyTorch张量并返回
    return (torch.LongTensor(enc_inputs), 
            torch.LongTensor(dec_inputs), 
            torch.LongTensor(dec_outputs))


# ==================== 自定义数据集类 ====================
class MyDataSet(Data.Dataset):
    """
    自定义的PyTorch数据集类
    
    这个类继承自torch.utils.data.Dataset，用于：
    1. 封装训练数据，使其能够被DataLoader使用
    2. 支持批量加载和随机打乱
    3. 提供标准的数据集接口
    
    Args:
        enc_inputs: 编码器输入张量
        dec_inputs: 解码器输入张量  
        dec_outputs: 解码器期望输出张量
    """
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs    # 存储编码器输入
        self.dec_inputs = dec_inputs    # 存储解码器输入
        self.dec_outputs = dec_outputs  # 存储解码器期望输出

    def __len__(self):
        """
        返回数据集的大小（样本数量）
        
        Returns:
            int: 数据集中的样本数量
        """
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        """
        根据索引获取一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (编码器输入, 解码器输入, 解码器期望输出)
        """
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
