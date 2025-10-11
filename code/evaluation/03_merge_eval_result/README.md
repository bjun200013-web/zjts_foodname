# LLM语义评估系统使用说明

## 概述

本系统使用多个大语言模型（Gemini、GPT-4o、DeepSeek、Claude）对菜名识别结果进行语义相似度评估。支持并行处理和自动结果合并，高效处理大规模评估任务。

## 核心文件

- `main.py`: 单批次评估脚本，调用多个LLM API进行评分
- `parallel.sh`: 并行评估启动脚本，将任务分割为多个批次并行执行
- `merge.py`: 结果合并脚本，整合所有批次结果并输出最终评分

## 完整并行评估流程

**输入**: 上一步模型推理的输出文件`/path/to/model/output.csv`（包含模型预测结果和ground truth的Excel文件）。

**使用方法**: `sh parallel.sh /path/to/model/output.csv`

**输出**：打印评估分数统计，以及保存LLM评分汇总表