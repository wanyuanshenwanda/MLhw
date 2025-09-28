# GitHub PR 分析实验

本项目实现了对单个给定数据集 GitHub 仓库 Pull Request (PR) 数据的特征工程以及爬取其他仓库的PR、模型训练与结果分析，涵盖如下两项任务：

1. **任务一：预测 PR 处理时间（Time-to-Close, 小时）**  
2. **任务二：预测 PR 是否合并（merged 二分类）**

整个流程关于给定数据集的训练主要通过 src/pipeline.py 完成，包括特征整合、按时间划分训练/测试集、基线模型训练、特征消融实验以及指标可视化。
	爬取数据包括GithubPR中的pull.py,new_data_project/src/data_preprocessing中的代码，爬取数据的训练代码主要位于new_data_project/src
	下面是关于给定数据集的训练介绍，爬取数据的训练介绍详细请查看new_data_project/README.md

## 数据与时间切分

- 所有原始文件位于 data/ 目录，主要使用：
  - PR_info_add_conversation.xlsx（PR 核心属性，含 created_at、closed_at 等时间字段）；
  - PR_features.xlsx（代码、文本、评论等统计特征）；
  - author_features.xlsx、reviewer_features.xlsx、project_features.xlsx（作者/评审/项目级特征）。
  - GitHubPR/data：
- 处理流程会生成聚合后的特征表 outputs/feature_table.csv，便于复用。
- 为避免数据泄漏，训练/测试按创建时间的先后顺序划分：
  - 训练集：created_at < 2021-06-01
  - 测试集：created_at >= 2021-06-01
  
  此外为了避免模型提前知道信息对训练数据做出一定处理



## 特征工程概要

- 文本/关键词：title/body 词数、字符长度、可读性评分以及 has_test/feature/bug/improve/document/refactor 等布尔标记。
- 代码变更规模：lines_added/deleted、segs_added/deleted/changed、files_added/deleted/changed、modify_proportion、test_churn 等。
- 结构：directories、language_types、file_types。
- 作者：author_* 系列特征与历史 PR 数（prev_prs）。
- 项目：project_* 系列聚合指标（周贡献、复审轮次等）。
- 评审：reviewer_* 系列指标。

清洗策略：统一时区转换、去除 inf、对缺失值使用训练集中位数填充（若全为空则回退为 0）。

## 环境准备

pip install -r requirements.txt，Python版本不限（建议3.10及以上），自行配置Github token为系统变量，建议使用conda虚拟环境

## 运行实验

python -m src.pipeline

运行完成后将在 outputs/ 目录得到：

- feature_table.csv：最终整合后的特征数据；
- regression_metrics.csv / classification_metrics.csv：测试集指标；
- regression_feature_ablation.csv / classification_feature_ablation.csv：特征消融结果；
- figures/：各指标与消融的可视化 PNG 图。

## 模型与指标

### 任务一：时间预测（单位：小时）

| 模型 | MAE | RMSE | R² |
| --- | --- | --- | --- |
| LinearRegression | 1325.82 | 1797.84 | -2.35 |
| RandomForest | 542.29 | 1142.75 | -0.35 |
| GradientBoosting | 575.84 | 1436.11 | -1.14 |
| HistGradientBoosting | 471.69 | 1089.13 | -0.23 |
| GradientBoosting + log1p(y) | 212.92 | 976.13 | 0.01 |

> 通过对目标变量使用 log1p 变换，梯度提升模型的 MAE 约为 212 小时（≈8.8 天），显著优于未变换的版本。

### 任务二：合并预测

| 模型 | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
| --- | --- | --- | --- | --- |
| LogisticRegression | 0.878 | 0.846 | 0.676 | 0.718 |
| RandomForest | 0.888 | 0.867 | 0.701 | 0.747 |
| GradientBoosting | 0.883 | 0.804 | 0.752 | 0.773 |

> 梯度提升分类器在 Macro-F1 上最优，整体准确率约 88%。

## 特征消融洞察

- 时间预测：移除 *author_* 或 *project_* 特征会导致误差显著上升（MAE +~9 小时）；文本/结构特征影响较小。说明作者与项目历史是预测 PR 处理时长的关键因素。
- 合并预测：去掉作者画像后，Macro-F1 从 0.77 降至 0.62，准确率也下降到 0.81，显示作者历史对合并概率判断影响最大。移除文本、结构等对结果影响相对有限。

详细数值见 outputs/*_feature_ablation.csv。

