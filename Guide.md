# 论文写作指导手册：MERT 与共形预测的结合

> 本手册用于指导 AI Agent 协助完成论文《Quantifying with Confidence: Conformal Prediction for Reliable Musical Style Similarity Ranking》（暂定名）。  
> 目标：将 **MERT**（大规模自监督音乐表示学习模型）与 **共形预测（Conformal Prediction, CP）** 结合，构建带统计置信度的音乐风格相似度评估框架。

---

## 1. 项目概述

- **研究问题**：现有音乐风格相似度评估基于自监督嵌入（如 MERT）输出点估计（单一余弦相似度分数），但缺乏可靠性度量。如何为每个预测提供具有理论保证的置信区间或预测集？
- **核心贡献**：
  - 首次将共形预测引入基于自监督音频嵌入的音乐风格相似度评估领域，实现不确定性量化。
  - 构建端到端框架：音频输入 → MERT 嵌入提取 → 风格相似度计算 → 共形预测校准 → 输出可信排名/区间。
  - 实验验证覆盖率和紧致性，展示在 AI 生成音乐过滤和采样中的应用优势。
- **关键词**：音乐风格、相似度、共形预测、不确定性量化、自监督学习、MERT、音频表示学习

---

## 2. 论文整体结构（拟）

| 章节 | 标题 | 篇幅建议 |
|------|------|----------|
| 1    | 引言 | 约 2 页 |
| 2    | 相关工作 | 约 2–3 页 |
| 3    | 方法 | 约 4 页 |
| 4    | 实验设计与结果 | 约 4–5 页 |
| 5    | 讨论 | 约 1 页 |
| 6    | 结论与展望 | 约 1 页 |
|      | 参考文献 | 按需 |

---

## 3. 各章节详细写作指南

### 3.1 引言 (Introduction)

- **开门见山**：指出音乐风格相似度评估是 MIR 和 AI 音乐生成的核心问题。
- **现有不足**：自监督音频嵌入方法（如 MERT）虽能有效捕获音色、和声、节奏等高层音乐信息，但相似度输出为单一标量分数，无法量化不确定性；在实际应用中（如过滤/采样生成样本），误判风险高。
- **解决方案**：引入共形预测，为每个相似度预测提供统计保证的置信集（confidence set），并定义覆盖率和紧致性指标。
- **论文贡献**：
  1. 首次将共形预测应用于自监督音乐表示学习的风格相似度评估。
  2. 提出基于 MERT 的共形预测框架，兼容任意音频嵌入方法。
  3. 通过实验证明该框架在保持排名准确性的同时，提供了可靠的不确定性估计，且能有效提升生成样本筛选与采样的可靠性。
- **结构预告**：简要说明后续章节安排。

> **Agent 任务**：初稿生成后，检查是否明确点出“不确定性”痛点，并突出 CP 与 MERT 结合的必要性。

---

### 3.2 相关工作 (Related Work)

- **音乐相似度与风格评估**：回顾基于手工特征、压缩距离、深度嵌入的方法；重点介绍 MERT（Wav2Vec2/BERT-like 架构、MLM 自监督训练、RVQ-VAE 声学教师 + CQT 音乐教师）。
- **自监督音频表示学习**：综述 Wav2Vec 2.0、HuBERT、data2vec、CLAP、Jukebox 等音频表示；对比它们在音乐风格捕获方面的能力。
- **共形预测**：简述原理（归纳共形预测、非一致性分数、p 值、预测集）；强调其在分类、回归、排序中的应用，并提及在生成模型评估中的新进展。
- **不确定性量化在 MIR 中的现状**：指出目前极少有工作将共形预测用于音乐风格相似度，本工作填补空白。
- **与现有生成评估方法对比**：对比 log-likelihood、FAD（Fréchet Audio Distance）、KID 等，指出它们同样缺乏逐样本置信度量。

> **Agent 任务**：收集相关文献（至少 15 篇），按主题整理；确保引用格式统一（如 IEEE 或 ACM 格式）。

---

### 3.3 方法 (Methodology) —— **核心章节**

#### 3.3.1 问题定义

- 给定目标风格语料库 `C = {C₁,…,Cₙ}`（一组音频文件）和待测样本 `x`（音频），通过 MERT 编码器提取嵌入向量，然后计算相似度 `S(x, C)`（标量，如余弦相似度均值）。
- 期望输出：对于预先设定的置信度 `1-α`（如 95%），生成一个预测集 `Γ(x) ⊆ ℝ`（或排名集合），使得 `P( y_true ∈ Γ(x) ) ≥ 1-α`，其中 `y_true` 为真实相似度（或真实排名）。

#### 3.3.2 MERT 概述（简洁）

- **架构**：基于 Wav2Vec 2.0 的 BERT-like Transformer 编码器。
  - MERT-v1-95M：12 层 Transformer，嵌入维度 768，95M 参数，预训练于 20K 小时音乐数据。
  - MERT-v1-330M：24 层 Transformer，嵌入维度 1024，330M 参数，预训练于 160K 小时音乐数据（**推荐**）。
- **预训练**：掩码语言建模（MLM）范式，使用 RVQ-VAE 声学教师和 CQT 音乐教师作为伪标签。
- **嵌入提取**：给定输入音频 `x`（重采样至 24kHz），通过 MERT 编码器获得多层隐藏状态 `h₁, h₂, …, h_L`（L = 13 对 95M，L = 25 对 330M），每层形状为 `[T, d]`。
  - 嵌入表示：`E_x = mean_pool(h_layer)`，选择最佳层或加权组合（不同层在不同任务上表现各异）。
  - 默认推荐：使用最后一层或中间若干层的均值池化。
- **相似度计算**：`S(x, C) = (1/|C|) Σ_{c∈C} cos(E_x, E_c)`，即待测样本嵌入与风格语料库中所有样本嵌入的余弦相似度均值。

#### 3.3.3 共形预测框架设计（重点）

- **校准集 (Calibration set)**：从 `C` 中预留一部分样本（如 20%）作为校准数据，其余作为参考集（用于计算相似度）。
- **非一致性分数 (Nonconformity score)**：定义 `α(x, y)` 衡量样本 `x` 与假设标签 `y` 的不一致程度。
  *推荐设计*：`α(x, y) = | S(x, C) - y |`，其中 `y` 为候选相似度值。
  *备选方案*：归一化 `α(x, y) = | S(x, C) - y | / σ_calib`。
- **计算 p 值**：`p(x, y) = ( |{ i : α_i ≥ α(x, y) }| + 1 ) / (m+1)`，其中 `m` 为校准集大小。
- **构建预测集**：`Γ(x) = { y : p(x, y) > α }`。
- **针对排名任务**：可构造排序预测集，使用成对共形预测或标签投票策略。

#### 3.3.4 实现细节

- 代码语言：Python 3.11 + PyTorch 2.7+ + HuggingFace Transformers。
- MERT 加载：`AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)`（一行代码，自动下载权重）。
- 共形预测实现：使用 `scikit-learn-contrib/MAPIE`（1.6k star）或自行实现归纳共形预测。
- 显著性水平 α 取 0.01, 0.05, 0.1。

> **Agent 任务**：为每个公式提供 LaTeX 代码；协助编写伪代码（算法框）；确保数学符号清晰连贯。

---

### 3.4 实验 (Experiments)

#### 3.4.1 数据集

- **风格数据集**：
  - **GTZAN**（1000 首 30 秒音频，10 流派）——经典 MIR 风格分类基准，公开可用。
  - **FMA (Free Music Archive)**（8000+ 首，8 大流派）——大规模公开数据集。
  - **MagnaTagATune**（25863 首片段，188 标签）——多标签风格/情绪标注。
- **生成样本**：使用 MusicGen / AudioLDM 2 / Suno 等模型生成音频样本，作为测试集评估风格过滤效果。
- 数据预处理：统一采样率至 24kHz（MERT 原生采样率），单声道转换。

#### 3.4.2 评估指标

- **覆盖率 (Coverage)**：预测集包含真实排名/相似度的比例，理想情况应接近 `1-α`。
- **平均预测集大小 (Average set size)**：衡量紧致性，越小越精确。
- **排名准确性**：Kendall τ 评估预测集与真实排名的吻合度。
- **过滤任务评估**：设定相似度阈值，比较原始相似度过滤与“置信下限过滤”的精确率/召回率。
- **采样任务评估**：使用置信下限指导采样，评估生成样本质量 FAD/KID。

#### 3.4.3 基线方法

- 原始 MERT 嵌入 + 余弦相似度（无置信度量）。
- 朴素 Bootstrap 区间估计（1000 次重采样）。
- 基于高斯近似的置信区间。
- 其他自监督音频嵌入 + CP：CLAP、Wav2Vec 2.0 嵌入作为对比。
- 手工音频特征（MFCC、chroma、mel-spec）作为传统基线。

#### 3.4.4 实验设置

- 校准集大小：分别取 10%、20%、30% 的 `C`。
- 随机种子固定，重复 10 次报告均值和标准差。
- 显著性水平 α 取 0.01, 0.05, 0.1。
- MERT 嵌入层选择：对比不同层（第 1/6/12/13 层等）的嵌入质量。

#### 3.4.5 预期结果（假设）

- 覆盖率接近名义水平（如 95% 置信度下覆盖率 ~94%-96%）。
- 平均预测集大小适中（例如相似度区间宽度 ~0.12-0.15）。
- 置信下限过滤显著降低假阳性率，同时保持较高召回率。
- CP 覆盖率稳定性优于 Bootstrap（不受分布假设影响）。
- MERT 嵌入在风格捕获上显著优于传统手工特征。

> **Agent 任务**：编写实验代码骨架；生成模拟数据辅助调试；提供绘图脚本。

---

### 3.5 讨论 (Discussion)

- 解释 valid coverage 的理论基础（交换性假设，在 i.i.d. 校准集下成立）。
- 分析预测集大小影响因素：MERT 嵌入空间的区分度、校准集规模、风格定义的粒度。
- CP vs Bootstrap：CP 轻量且无分布假设，Bootstrap 昂贵且无理论保证。
- 局限：校准集需与测试集同分布；MERT 主要捕获声学特征，高层文化/历史风格信息有限；共形预测输出为集合而非单一区间。
- 未来工作：多模态风格（歌词+音频）、动态校准、交互式评估场景。

> **Agent 任务**：起草讨论段落，强调理论优势和实践挑战。

---

### 3.6 结论与展望 (Conclusion)

- 总结：成功将共形预测与 MERT 集成，提供统计保证的音乐风格相似度评估。
- 应用价值：为 AI 音乐生成评估提供可靠工具，支持可信赖的自动评估流水线。
- 展望：更大规模 SSL 模型、端到端风格感知嵌入、实时交互式评估。

---

## 4. 写作风格与格式要求

- **语言**：英文（符合学术论文标准），现在时描述方法，过去时描述实验。
- **数学符号**：LaTeX，所有变量、矩阵、分布均需定义清晰。
- **图表**：至少包括：
  - CP 算法伪代码
  - MERT 嵌入提取流程示意图
  - 覆盖率 vs. α 折线图
  - 预测集大小分布箱线图
  - 过滤任务 PR 曲线
- **引用**：BibTeX，IEEE/ACM 格式。优先引用 MERT 原文（Li et al., ICLR 2024）、共形预测经典文献（Vovk et al.）、CP 综述（Angelopoulos & Bates, 2023）。
- **字数**：正文（不含参考文献）约 8000–10000 词。

---

## 5. Agent 协作流程

### 5.1 阶段划分与里程碑

| 阶段 | 任务 | 交付物 | 状态 |
|------|------|--------|------|
| 1 | 文献调研 & 引言撰写 | 引言初稿 + 参考文献列表 | ✅ `paper/sections/01_introduction.tex` + `references/refs.bib` |
| 2 | 方法详细设计 | 数学模型、伪代码、实现说明 | ✅ 方案A(CP分类)已确定 |
| 3 | 实验代码与数据准备 | 可运行实验脚本 + 数据 | ✅ 全部实现 |
| 4 | 实验运行与结果分析 | 图表、表格、结果解读 | ✅ CP+Bootstrap+Gaussian 对比完成 |
| 5 | 全文整合与润色 | 完整初稿、语法检查、格式修正 | ✅ `paper/main.tex` + `paper/main.pdf` (19 页, IEEE) |
| 6 | FMA 补充数据集验证 | FMA 嵌入提取 + CP 实验 + 对比图 | ✅ 7994/8000 文件，8 流派，跨数据集覆盖保证验证通过 |

### 5.2 Agent 角色与职责

- **文献收集**：按关键词检索论文，提供摘要和引用。
- **LaTeX 辅助**：生成公式、表格、图表代码。
- **代码助手**：编写实验脚本（Python + PyTorch + HuggingFace + MAPIE）。
- **写作助手**：按大纲生成段落，检查逻辑连贯性和术语一致性。
- **审校**：检查拼写、语法、标点，确保学术风格。

### 5.3 交互规则

- 用户提出具体问题或需求，Agent 返回内容、建议或代码。
- 不确定内容需明确标注并请求用户确认。
- 每阶段末汇总进展，调整计划。

---

## 6. 关键参考文献（种子列表）

1. Li, Y. et al. (2023). MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training. *ICLR 2024*. (arXiv:2306.00107)
2. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
3. Angelopoulos, A. N. & Bates, S. (2023). Conformal Prediction: A Gentle Introduction. *Foundations and Trends in ML*.
4. Baevski, A. et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *NeurIPS 2020*.
5. Hsu, W.-N. et al. (2021). HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. *TASLP*.
6. Yang, L. C. & Lerch, A. (2018). On the Evaluation of Generative Models in Music. *Neural Computing and Applications*.
7. Castellon, R., Donahue, C. & Liang, P. (2021). Codified audio language modeling learns useful representations for music information retrieval. *ISMIR 2021*.
8. Cordier, T. et al. (2023). Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE library. *COPA 2023*.
9. Défossez, A. et al. (2022). High Fidelity Neural Audio Compression. *arXiv:2210.13438*. (EnCodec)
10. Tzanetakis, G. & Cook, P. (2002). Musical genre classification of audio signals. *IEEE TASLP*. (GTZAN)
11. 其他 CP + MIR/音频/生成评估最新进展（待补充）。

---

## 7. 附加资源

- **代码仓库**：
  - MERT HuggingFace：<https://huggingface.co/m-a-p/MERT-v1-330M>（一行 `AutoModel.from_pretrained` 加载）
  - MERT GitHub：<https://github.com/yizhilll/MERT>（481 star）
  - MAPIE（CP）：<https://github.com/scikit-learn-contrib/MAPIE>（`pip install mapie`）
  - 项目私有仓库：包含嵌入提取、相似度计算、CP 校准模块、实验脚本
- **数据集下载**：
  - GTZAN：<https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification>
  - FMA：<https://github.com/mdeff/fma>
- **预训练模型**：MERT-v1-330M 通过 HuggingFace Hub 自动下载，无需手动管理。
- **实验环境**：Python 3.11, PyTorch 2.7+, transformers, torchaudio, scikit-learn, numpy, matplotlib, pandas, mapie

---

## 8. 时间线与风险控制

- **总工期**：建议 4–6 周（全职）。
- **风险**：
  - GTZAN（999 有效文件/10 流派）数据量较小，风格多样性有限 → ✅ 已补充 FMA Small（7994/8000，8 流派），实验已完成。
  - ~~共形预测在排序任务中的非一致性分数设计可能不收敛~~ → 已改用方案A(CP分类)，以流派原型质心替代。
  - MERT 嵌入层选择（不同层在不同任务上效果不同）→ 当前仅用最后一层，待后续验证其他层。
  - 覆盖率达标 ✅：CP 在两个数据集上均紧贴名义值（GTZAN: 0.993/0.955/0.910; FMA: 0.988/0.948/0.898）。
  - FMA 嵌入区分度低（δ=0.026 vs GTZAN δ=0.044）→ 预测集偏大(6-8/8)，反映真实世界流派标签模糊性，CP 诚实输出而非虚假高置信。
- **应对**：每周 review 进度，及时调整。

---

**最后提醒**：本手册是动态文档，随着研究深入需不断更新。Agent 应定期询问用户是否需要修订大纲或补充细节。

---

*最后更新：2026-07-25 | 阶段：全文初稿 + PDF 编译完成 + FMA 补充数据集实验完成 (21/23 步骤) | Python 3.11 | PyTorch 2.7+*
