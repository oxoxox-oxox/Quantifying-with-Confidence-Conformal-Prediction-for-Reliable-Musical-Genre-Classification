# Agent Skill: MERT + Conformal Prediction 论文写作助手

> 核心指令：本文档是 Agent 行为基准。每次对话开始时读取此文件，无需重复读取 Guide.md。
> 用户指令优先于本文档。

---

## 1. 项目一句话

将 **MERT**（Music undERstanding model with large-scale self-supervised Training, ICLR 2024）与 **共形预测（Conformal Prediction）** 结合，构建带统计置信度的音乐风格相似度评估框架，产出 8000–10000 词英文学术论文。

**关键词**：音乐风格、相似度、共形预测、不确定度量化、自监督学习、MERT、音频表示学习

---

## 2. Agent 角色

| 角色 | 职责 |
|------|------|
| **文献收集** | 按关键词检索论文，提供摘要与引用 |
| **LaTeX 辅助** | 生成公式、表格、算法伪代码、图表代码 |
| **代码助手** | 编写/调试实验脚本（Python 3.11 + PyTorch + HuggingFace + MAPIE） |
| **写作助手** | 按大纲生成段落，检查逻辑连贯性与术语一致性 |
| **审校** | 检查拼写、语法、标点，确保学术风格 |

---

## 3. 写作规范

- **语言**：英文。方法用现在时，实验用过去时。
- **数学**：LaTeX。所有变量、矩阵、分布均需定义清晰。
- **引用**：BibTeX，IEEE 格式（`ieeetr`）。优先引用种子文献（§8）。
- **图表**：至少含 MERT 嵌入提取流程图、CP 算法伪代码、覆盖率 vs α 折线图、预测集大小分布箱线图、过滤任务 PR 曲线。

---

## 4. 核心术语

| English | 中文 | 说明 |
|---------|------|------|
| MERT | MERT | Music undERstanding model with large-scale self-supervised Training (ICLR 2024) |
| MERT-v1-95M | MERT 95M | 12 层 Transformer, 768 dim, 95M 参数, 20K 小时预训练 |
| MERT-v1-330M | MERT 330M | 24 层 Transformer, 1024 dim, 330M 参数, 160K 小时预训练 **（推荐）** |
| HuggingFace Transformers | HF Transformers | 一行 `AutoModel.from_pretrained` 加载 MERT 预训练权重 |
| Conformal Prediction (CP) | 共形预测 | 提供统计保证的置信预测框架 |
| Nonconformity score | 非一致性分数 | `α(x, y)` 衡量样本与候选标签不一致程度 |
| Calibration set | 校准集 | 用于计算经验分位数的预留数据 |
| Coverage | 覆盖率 | 预测集包含真实标签的比例 |
| Average set size | 平均预测集大小 | 衡量预测集紧致性 |
| Prediction set / Confidence set | 预测集 / 置信集 | `Γ(x)`，满足覆盖保证的候选标签集合 |
| p-value | p 值 | 用于判定候选标签是否纳入预测集 |
| Significance level (α) | 显著性水平 | 取 0.01, 0.05, 0.1 |
| GTZAN | GTZAN | 1000 首/10 流派/30s 每首，经典 MIR 风格分类基准 |
| FMA | Free Music Archive | 8000+ 首/8 大流派，大规模公开数据集 |
| MLM | 掩码语言建模 | Masked Language Modeling, MERT 的预训练范式 |
| RVQ-VAE / CQT | 残差向量量化 VAE / 常数 Q 变换 | MERT 的声学教师和音乐教师 |

---

## 5. 交互规则

1. 不确定内容必须明确标注并请求用户确认。
2. 每阶段结束时汇总进展，提请用户 review。
3. 优先遵循用户当前指令，本文档为默认指导。
4. 写代码前先搜索现有代码库，遵循已有风格与库。
5. 不主动 commit/PR；不生成文档文件（除非用户明确要求）。

---

## 6. 论文结构速查

| 章节 | 标题 | 核心要点 |
|------|------|----------|
| 1 | Introduction | 痛点：SSL 嵌入缺乏不确定性量化 → CP 解决方案 → 三点贡献 |
| 2 | Related Work | 音乐相似度(含 MERT/SSL)、CP 原理、MIR UQ 现状、生成评估对比 |
| 3 | Methodology | 问题定义 → MERT 嵌入概述 → CP 框架设计 → 实现细节 |
| 4 | Experiments | 数据集(GTZAN 999/1000)、指标(Coverage/Set size)、基线(Bootstrap/Gaussian 已实现)、结果(CP cover紧贴名义值, Gaussian在α=0.01处不足) |
| 5 | Discussion | 覆盖保证成立(0.993/0.955/0.910)、CP vs Bootstrap(大样本趋同/小样本CP有保证) vs Gaussian(厚尾假设不成立)、局限(预测集偏大 5.4-8.5)、未来工作 |
| 6 | Conclusion | 成果总结、应用价值、展望 |

---

## 7. 阶段与里程碑

| 阶段 | 任务 | 交付物 | 状态 |
|------|------|--------|------|
| 1 | 文献调研 & 引言 | 引言初稿 + 参考文献列表 | ✅ |
| 2 | 方法详细设计 | 数学模型、伪代码、实现说明 | ✅ |
| 3 | 实验代码与数据 | 可运行脚本 + 模拟结果 | ✅ |
| 4 | 实验运行与分析 | 图表、表格、结果解读 | ✅ |
| 5 | 全文整合与润色 | 完整初稿 + PDF | ✅ |

**当前阶段**：全文初稿 + PDF 编译完成。已完成 21/23 步骤 (1-19, 20-23, 除 11-12)。

## 进度记录

| 步骤 | 任务 | 状态 | 备注 |
|------|------|------|------|
| 1 | 基础搭建 | ✅ | 目录结构 + requirements.txt + 环境验证 |
| 2 | MERT 验证 | ✅ | 模型加载成功（`m-a-p/MERT-v1-330M`），GPU 推理通过 |
| 3 | 数据准备 | ✅ | GTZAN 验证通过（1000 WAV, 10 流派），1 文件损坏(jazz.00054.wav) |
| 4 | 文献调研 | ✅ | 20 篇文献 → `references/refs.bib`, 覆盖 MERT/CP/生成/数据集/多模态 5 个领域 |
| 5 | 嵌入提取 | ✅ | 999 文件 → (999, 1024), ~9 min, `outputs/embeddings/embeddings.npy` |
| 6 | 嵌入验证 | ✅ | NaN/Inf=0, intra=0.917, inter=0.873, delta=0.044 |
| 7 | 风格标注 | ✅ | 10 流派, jazz=99, 其余=100, `labels.csv` + `label_ids.npy` |
| 8 | CP 分类实现 | ✅ | 方案A: 流派原型质心 + 非一致性分数 α=1-cos(E_x, centroid) |
| 9 | CP 实验运行 | ✅ | 5-fold 分层, 60/20/20 split, coverage 紧贴名义值 |
| 10 | 基线对比 | ✅ | Bootstrap 0.987, Gaussian 0.965/0.932/0.907; CP 优于或等于两者 |
| 11 | 过滤实验 | ⏳ | 待实现：基于置信下限的生成样本过滤 |
| 12 | 采样实验 | ⏳ | 待实现：置信下限引导的生成采样 |
| 20 | FMA 数据准备 | ✅ | `scripts/prepare_fma.py --download`，7994/8000 可用的 8 流派，每类 ~1000 |
| 21 | FMA 嵌入提取 | ✅ | `experiments/extract_embeddings_fma.py`，7994 → (7994, 1024)，~68 min |
| 22 | FMA CP 实验 | ✅ | `experiments/cp_fma_experiment.py`，CP+Bootstrap+Gaussian，intra=0.847, delta=0.026 |
| 23 | FMA 对比图 | ✅ | `experiments/plot_fma_comparison.py`，3 张对比图 |
| 13 | 图表生成 | ✅ | 3张: coverage_vs_alpha, set_size_boxplot, coverage_deviation |
| 14 | 引言 | ✅ | `paper/sections/01_introduction.tex`, ~1000 词, 含贡献声明 + 实验数据 |
| 15 | 相关工作 | ✅ | `paper/sections/02_related_work.tex`, ~180 行, 5 个子节, 覆盖 SSL/MERT/CP/UQ/评估
| 16 | 方法 | ✅ | `paper/sections/03_methodology.tex`, 6 个子节: 问题定义+嵌入提取+CP框架+实现+基线
| 17 | 实验撰写 | ✅ | `paper/sections/04_experiments.tex`, 6 个子节: 数据集+设置+嵌入质量+覆盖率+集大小+总结, 含表格结果
| 18 | 讨论+结论 | ✅ | `paper/sections/05_discussion.tex` + `paper/sections/06_conclusion.tex`
| 19 | 最终整合 | ✅ | `paper/main.tex` (19 页, ~630 KB), abstract + 5 图嵌入 + 参考文献 (ieeetr)

### 环境快照

| 组件 | 版本 |
|------|------|
| Python | 3.11.15 |
| conda env | `essay_env` |
| PyTorch | 2.7.1+cu128 |
| GPU | RTX 5070 Laptop 8GB |
| torchaudio | 2.7.1+cu128 |
| transformers | 5.14.1 |
| scikit-learn | 1.9.0 |
| MAPIE | 1.4.1 |
| MERT model | `m-a-p/MERT-v1-330M` (HF cache) |

### 待办

- 论文全文初稿 + PDF 编译已完成（步骤 1-10, 13-19）
- FMA 补充数据集实验已完成（步骤 20-23），8 流派/7994 有效文件，CP 覆盖保证跨数据集验证通过
- 剩余待实现：过滤实验（步骤 11）、采样实验（步骤 12）

---

## 8. 关键引用种子

1. Li, Y. et al. (2023). MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training. *ICLR 2024*.
2. Vovk, V. et al. (2005). *Algorithmic Learning in a Random World*. Springer.
3. Angelopoulos, A. N. & Bates, S. (2023). Conformal Prediction: A Gentle Introduction. *Foundations and Trends in ML*.
4. Baevski, A. et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *NeurIPS*.
5. Hsu, W.-N. et al. (2021). HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. *TASLP*.
6. Yang, L. C. & Lerch, A. (2018). On the Evaluation of Generative Models in Music. *Neural Computing and Applications*.
7. Cordier, T. et al. (2023). Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE library. *COPA 2023*.
8. Défossez, A. et al. (2022). High Fidelity Neural Audio Compression. *arXiv:2210.13438*. (EnCodec)
9. Tzanetakis, G. & Cook, P. (2002). Musical genre classification of audio signals. *IEEE TASLP*. (GTZAN)
10. 其他 CP + MIR/音频/生成评估最新进展（待补充）。

---

## 9. 技术栈

| 组件 | 版本/工具 |
|------|-----------|
| Python | 3.11 |
| PyTorch | 2.7+ (CUDA 12.8) |
| torchaudio | 2.7+ |
| HuggingFace Transformers | 最新稳定版 |
| scikit-learn | 1.9+ |
| NumPy, Matplotlib, Pandas | 最新稳定版 |
| MAPIE | 1.4+ (`pip install mapie`) |
| MERT 模型 | `m-a-p/MERT-v1-330M` (HF Hub, 自动下载) |
| 排版 | LaTeX (BibTeX, IEEE/ACM 模板) |

---

## 10. 方法速查（§3）

**核心链路**：音频输入 → MERT Encoder 提取嵌入 `E_x` → 余弦相似度 `S(x,C)` → 校准集计算非一致性分数 → p 值 → 预测集 `Γ(x)`

**MERT 加载**：
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
```

**嵌入提取**（分段策略：10s 窗口，5s 重叠）：
```python
outputs = model(audio_input, output_hidden_states=True)
# hidden_states: list of [1, T, 1024], 25 layers for 330M
E_x = outputs.hidden_states[-1].mean(dim=1)  # [1, 1024]
```

**CP 分类方案（方案A）**：
- 流派原型质心：对参考集中每个流派的所有嵌入求均值 → `centroid_k`
- 非一致性分数：`α(x, k) = 1 - cos(E_x, centroid_k)`（距离越大越不一致）
- p 值：`p(x, k) = (|{ i ∈ cal : α_i ≥ α(x, k) }| + 1) / (|cal| + 1)`
- 预测集：`Γ(x) = { k : p(x, k) > α }`
- 校准集比例：20%

**基线对比（§4 实验部分）**：
- Bootstrap: 重采样 1000 次估计分位数，大样本下接近 CP，但无有限样本保证
- Gaussian: 假设非一致性分数 ~ N(μ,σ²)，高置信度(α=0.01)时覆盖率不足(0.965 vs 0.99)
- **CP 优势**：分布无关、有限样本覆盖保证，在音乐风格这种厚尾分布任务中与 Gaussian 形成对比

**实验结果（GTZAN, 5-fold, 60/20/20 split）**：

| α | 名义覆盖率 | CP | Bootstrap | Gaussian |
|---|-----------|-----|-----------|----------|
| 0.01 | 0.99 | 0.993 | 0.987 | 0.965 |
| 0.05 | 0.95 | 0.955 | 0.950 | 0.932 |
| 0.10 | 0.90 | 0.910 | 0.910 | 0.907 |

**FMA 结果 (7994 首, 8 流派, 5-fold, 60/20/20 split)**：

| α | 名义覆盖率 | CP | Bootstrap | Gaussian |
|---|-----------|-----|-----------|----------|
| 0.01 | 0.99 | 0.988 | 0.987 | 0.966 |
| 0.05 | 0.95 | 0.948 | 0.947 | 0.934 |
| 0.10 | 0.90 | 0.898 | 0.896 | 0.903 |

**跨数据集对比**：

| 指标 | GTZAN (999) | FMA (7994) |
|------|:----------:|:----------:|
| 流派数 | 10 | 8 |
| Intra-class cos | 0.917 | 0.847 |
| Inter-class cos | 0.873 | 0.821 |
| Separation delta | 0.044 | 0.026 |
| CP avg set size (α=0.05) | 6.69 | 6.85 |

- FMA 嵌入区分度显著低于 GTZAN (delta 0.026 vs 0.044)，因为 8 个顶级流派间重叠度高
- CP 覆盖保证在两个数据集上均成立，验证了分布无关性质
- FMA 预测集偏大 (>6/8)，反映流派标签模糊性 —— 这是大规模真实数据集的固有特征，也验证了 CP 的诚实性

---

## 11. 运行命令速查

| 命令 | 用途 |
|------|------|
| `python scripts/verify_mert.py` | 验证 MERT 模型加载 + GPU 推理 |
| `python scripts/prepare_gtzan.py` | 验证 GTZAN 数据集结构 |
| `python experiments/extract_embeddings.py` | 提取 MERT 嵌入 (999/1000, ~9 min) |
| `python experiments/verify_embeddings.py` | 嵌入质量验证 + PCA 可视化 |
| `python experiments/cp_classification.py` | CP 分类实验 (5-fold) |
| `python experiments/baselines.py` | Bootstrap + Gaussian 基线对比 |
| `python experiments/plot_results.py` | 生成论文图表 (3 张) |
| `python scripts/prepare_fma.py --download` | 下载 FMA Small (7.2 GB) + 元数据 |
| `python experiments/extract_embeddings_fma.py` | 从 FMA 提取 MERT 嵌入 (~80 min) |
| `python experiments/cp_fma_experiment.py` | FMA CP 分类 + 基线对比 |
| `python experiments/plot_fma_comparison.py` | GTZAN vs FMA 对比图 |

---

*最后更新：2026-07-25 | 进度：21/23 步骤完成 | 全文初稿 + PDF 编译通过 (19 页) + FMA 补充实验完成 | 方案：MERT + Conformal Prediction (方案A) | Python 3.11 | PyTorch 2.7+*
