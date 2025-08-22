# Transformer模型训练主程序
# 参考资料: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
"""
这是Transformer模型的训练主程序，实现了完整的训练流程：
1. 数据准备和加载
2. 模型初始化
3. 损失函数和优化器设置
4. 训练循环
5. 模型保存

主要用于演示如何训练一个简单的序列到序列Transformer模型。
"""

import torch.nn as nn
import torch.optim as optim
from datasets import *
from transformer import Transformer

if __name__ == "__main__":
    
    # ==================== 数据准备阶段 ====================
    # 生成训练数据：源序列、目标输入序列、目标输出序列
    enc_inputs, dec_inputs, dec_outputs = make_data()
    
    # 创建数据加载器
    # batch_size=2: 每批处理2个样本
    # shuffle=True: 随机打乱数据顺序，提高训练效果
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

    # ==================== 模型初始化阶段 ====================
    # 创建Transformer模型并移到GPU
    model = Transformer().cuda()
    
    # 定义损失函数：交叉熵损失
    # ignore_index=0: 忽略填充符号(PAD)的损失计算
    # 这样模型就不会因为预测填充符号而受到惩罚
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 定义优化器：随机梯度下降(SGD)
    # lr=1e-3: 学习率为0.001
    # momentum=0.99: 动量参数，帮助加速收敛和减少震荡
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    # ==================== 训练循环阶段 ====================
    # 训练50个epoch（完整遍历数据集50次）
    for epoch in range(50):
        # 遍历每个批次的数据
        for enc_inputs, dec_inputs, dec_outputs in loader:
            # 数据格式说明：
            # enc_inputs : 源序列（编码器输入） [batch_size, src_len]
            # dec_inputs : 目标序列（解码器输入） [batch_size, tgt_len] 
            # dec_outputs: 目标序列（期望输出） [batch_size, tgt_len]

            # 将数据移到GPU进行计算
            enc_inputs = enc_inputs.cuda()
            dec_inputs = dec_inputs.cuda() 
            dec_outputs = dec_outputs.cuda()
            
            # 前向传播：通过模型得到预测结果和注意力权重
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # outputs: 模型预测的词汇分布 [batch_size * tgt_len, tgt_vocab_size]
            
            # 计算损失：比较预测结果和真实标签
            # dec_outputs.view(-1): 将目标序列展平为一维 [batch_size * tgt_len]
            loss = criterion(outputs, dec_outputs.view(-1))
            
            # 打印训练进度
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            
            # 反向传播和参数更新
            optimizer.zero_grad()  # 清零梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数
    
    # ==================== 模型保存阶段 ====================
    # 训练完成后保存模型
    torch.save(model, 'model.pth')
    print("模型训练完成并已保存到 model.pth")
