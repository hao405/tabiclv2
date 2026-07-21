## 1.写作定位

LLM的经验告诉我们，在推理的时候引入适当的computation可以有效地提高模型性能， 最近在tabular foundation model中也有不少researcher研究这个问题。但是如何为什么引入computation会有效以及如何更加有效滴设计finetuning依然是underexplored的

## 2.整体框架
1. introduction
2. related works
    2.1 tabular foundation model
    tabular foundation model是做什么的，最早的工作室xxx，有哪些方法，分成哪几类（看着最近的pfn论文写）
    2.2 finetuning for foundation model
    finetuning是什么，为什需要，最早的LLM的工作是xxx，有哪些方法，分成哪几类（这部分你多找论文，这儿估计你读的很不够）
3. Theory部分（主要是讲bound），具体organization可以参考我PAMI的论文
理论部分接着一段讨论（可以我来写）
4. 模型部分，根据bound的holdout-querymismatch项，提出一个模型设计思路（画图）
5. 实验