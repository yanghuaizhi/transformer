# 模型测试和推理模块 - 用于验证训练好的Transformer模型
# 参考资料: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
"""
这个模块用于测试训练好的Transformer模型，主要功能包括：
1. 实现贪婪解码算法进行序列生成
2. 加载训练好的模型进行推理
3. 展示翻译结果

贪婪解码是一种简单的序列生成策略，每次都选择概率最高的词作为下一个输出。
"""

from datasets import *

def test(model, enc_input, start_symbol):
    """
    使用贪婪解码算法进行序列生成
    
    贪婪解码的工作原理：
    1. 首先通过编码器处理源序列
    2. 从开始符号开始，逐步生成目标序列
    3. 每一步都选择概率最高的词作为下一个输出
    4. 重复直到生成完整序列
    
    Args:
        model: 训练好的Transformer模型
        enc_input: 编码器输入（源序列） [1, src_len]
        start_symbol: 开始符号的索引
        
    Returns:
        dec_input: 生成的目标序列 [1, tgt_len]
    """
    # 步骤1: 通过编码器处理源序列，获得上下文表示
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    
    # 步骤2: 初始化解码器输入序列（全零张量）
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    
    # 步骤3: 从开始符号开始生成
    next_symbol = start_symbol
    
    # 步骤4: 逐步生成目标序列
    for i in range(0, tgt_len):
        # 将当前符号放入解码器输入的第i个位置
        dec_input[0][i] = next_symbol
        
        # 通过解码器获得当前状态的输出
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        
        # 通过输出投影层得到词汇表上的分数
        projected = model.projection(dec_outputs)
        
        # 选择概率最高的词（贪婪策略）
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    
    return dec_input

# ==================== 模型测试主程序 ====================
if __name__ == "__main__":
    # 步骤1: 准备测试数据
    enc_inputs, dec_inputs, dec_outputs = make_data()
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
    
    # 步骤2: 获取一个测试样本
    enc_inputs, _, _ = next(iter(loader))
    
    # 步骤3: 加载训练好的模型
    model = torch.load('model.pth')
    
    # 步骤4: 使用贪婪解码生成翻译结果
    # 取第一个样本进行测试，reshape为 [1, src_len] 并移到GPU
    test_input = enc_inputs[0].view(1, -1).cuda()
    predict_dec_input = test(model, test_input, start_symbol=tgt_vocab["S"])
    
    # 步骤5: 通过完整模型获得最终预测
    predict, _, _, _ = model(test_input, predict_dec_input)
    
    # 步骤6: 获取预测的词汇索引（选择概率最高的词）
    predict = predict.data.max(1, keepdim=True)[1]
    
    # 步骤7: 展示翻译结果
    # 将索引转换回对应的词汇并打印
    source_words = [src_idx2word[int(i)] for i in enc_inputs[0]]  # 源语言句子
    target_words = [idx2word[n.item()] for n in predict.squeeze()]  # 翻译结果
    
    print(f"源句子: {' '.join(source_words)}")
    print(f"翻译结果: {' '.join(target_words)}")
    print(f"完整对比: {source_words} -> {target_words}")
