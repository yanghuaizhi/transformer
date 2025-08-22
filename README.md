# Transformer 模型实现

## 项目简介

Transformer是谷歌在2017年发表的论文《Attention Is All You Need》中提出的革命性模型架构。经过这些年的大量工业应用和学术验证，Transformer在深度学习领域已经占据了重要地位。许多知名模型如BERT、GPT等都是基于Transformer架构发展而来。

本项目以**中文翻译英文**为例，完整实现了Transformer模型的训练和推理过程，并提供了详细的中文注释，帮助初学者理解Transformer的核心原理和实现细节。

## 项目特色

- ✅ **完整实现**：包含Transformer的所有核心组件（编码器、解码器、注意力机制等）
- ✅ **详细注释**：每个文件都有完整的中文注释，解释核心逻辑和实现原理
- ✅ **简单易懂**：使用简单直白的语言，避免复杂术语，适合初学者学习
- ✅ **可运行示例**：提供完整的训练和测试代码，可以直接运行查看效果

## 文件结构

```
transformer/
├── transformer.py    # Transformer模型核心实现
├── main.py          # 模型训练主程序
├── datasets.py      # 数据处理和加载模块
├── test.py          # 模型测试和推理模块
└── README.md        # 项目说明文档
```

### 文件说明

- **transformer.py**: 包含Transformer模型的完整实现，包括：
  - 位置编码（PositionalEncoding）
  - 缩放点积注意力（ScaledDotProductAttention）
  - 多头注意力（MultiHeadAttention）
  - 前馈网络（PoswiseFeedForwardNet）
  - 编码器和解码器层
  - 完整的Transformer模型

- **main.py**: 模型训练主程序，包括：
  - 数据加载和预处理
  - 模型初始化
  - 训练循环和参数优化
  - 模型保存

- **datasets.py**: 数据处理模块，包括：
  - 训练数据定义
  - 中英文词汇表构建
  - 数据转换和加载

- **test.py**: 模型测试模块，包括：
  - 贪婪解码算法实现
  - 模型推理和翻译结果展示

## 快速开始

### 环境要求

- Python 3.6+
- PyTorch 1.0+
- CUDA（可选，用于GPU加速）

### 安装依赖

```bash
pip install torch
```

### 运行步骤

1. **训练模型**
   ```bash
   python main.py
   ```
   这将开始训练Transformer模型，训练过程中会显示每个epoch的损失值。训练完成后，模型会保存为`model.pth`文件。

2. **测试模型**
   ```bash
   python test.py
   ```
   这将加载训练好的模型，对测试数据进行翻译，并显示翻译结果。

## 训练数据

项目使用了3个简单的中英文对照句子作为训练数据：

- 我是学生 → I am a student
- 我喜欢学习 → I like learning
- 我是男生 → I am a boy

虽然数据量很小，但足以演示Transformer模型的完整训练和推理过程。

## 模型架构

Transformer模型采用编码器-解码器架构：

### 编码器（Encoder）
- 多层编码器层堆叠
- 每层包含多头自注意力和前馈网络
- 使用残差连接和层归一化

### 解码器（Decoder）
- 多层解码器层堆叠
- 每层包含掩码多头自注意力、编码器-解码器注意力和前馈网络
- 使用残差连接和层归一化

### 关键组件
- **注意力机制**：允许模型关注输入序列的不同位置
- **位置编码**：为序列中的每个位置添加位置信息
- **多头注意力**：并行计算多个注意力头，捕获不同类型的依赖关系

## 学习资源

- 原始论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 参考实现：[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- 详细解释：[知乎专栏文章](https://zhuanlan.zhihu.com/p/403433120)

## 注意事项

1. 本项目主要用于学习和理解Transformer原理，训练数据较少
2. 实际应用中需要更大的数据集和更复杂的数据预处理
3. 模型参数可以根据具体任务进行调整
4. 建议使用GPU进行训练以提高效率

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目采用MIT许可证。

