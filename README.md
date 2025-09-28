# GitHub PR ML 实验报告

本实验聚焦 GitHub 仓库中的 Pull Request（PR）数据，围绕两个核心任务展开：

- **任务一：预测 PR 处理时长** —— 本实现选择“从创建到关闭的时间（Time-to-Close, `time_to_close_hours`）”作为回归目标，同时保留 Time-to-First Response 等字段以便扩展。
- **任务二：预测 PR 是否被合并** —— 将 PR 的 `merged` 字段作为二分类标签，区分成功合并与关闭未合并的情况。

完整代码位于 `src/` 目录，包含数据整合、特征工程、模型训练、特征消融与可视化；`GithubPR/` 与 `new_data_project/` 提供数据扩展与多仓库实验支持。

## 1. 数据来源与防泄漏策略

### 1.1 基础数据

- 原始文件位于 `data/`：
  - `PR_info_add_conversation.xlsx`：PR 主体信息（创建/合并/关闭时间、标题、正文等）。
  - `PR_features.xlsx`：代码变更、文本、评论相关统计特征。
  - `author_features.xlsx`、`reviewer_features.xlsx`、`project_features.xlsx`：作者画像、评审画像与项目级聚合特征。
  - `PR_comment_info.xlsx` 等辅助文件用于补充首条评论时间等信息。
- 运行 `python -m src.pipeline` 会生成 `outputs/feature_table.csv`，以便复用与下游实验。

### 1.2 时间切分与数据泄漏控制

- 在 `src/pipeline.py` 中通过 `split_by_time` 函数按创建时间划分训练/测试集：
  - 训练集：`created_at < 2021-06-01`
  - 测试集：`created_at ≥ 2021-06-01`
- 切分前会先丢弃目标列的缺失项，确保评估公平；同时按时间排序保证模型只看到过去数据。
- 去除泄漏风险特征：
  - 在 `src/data_preparation.py` 中删除如 `time_to_merge_hours`、`first_comment_at` 等潜在泄漏字段，避免模型提前获知未来信息。

### 1.3 数据扩展与爬虫

- `GithubPR/pull.py` 提供基于 GitHub REST API 的采集脚本：
  - 需要在环境变量或 `.env` 中设置 `GITHUB_TOKEN`。
  - 支持分页抓取 PR 元数据、文件改动、评审记录，并内置特征含义说明。
  - 提供目录统计、关键字检测等示例函数，便于与本仓库特征体系对齐。
- `new_data_project/` 目录用于多仓库（如 vscode、pytorch、paddle、stockfish）实验，内含 `Makefile` 和 `src/` 代码，对多源数据进行清洗、整合与训练。
- 速率限制处理策略：
   github使用个人访问令牌 (PAT) 认证后，限额是每小时5000次请求。在处理分页时和爬取每个PR之间添加短暂延迟，避免过快请求
  ```time.sleep(0.5)```
- 相应的API
  - 获取指定仓库的所有PR（包括open和closed）```https://api.github.com/repos/{owner}/{repo}/pulls```
  - 获取单个PR的详细信息 ```https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}```
  - 获取单个PR修改的文件列表 ```https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files```
  - 获取单个PR的评审信息 ```https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews```
  - 获取仓库的贡献者列表，用于判断用户身份 ```https://api.github.com/repos/{owner}/{repo}/contributors```
  - 获取作者在此仓库中之前的PR数量 ```https://api.github.com/search/issues ```需要添加一些参数

## 2. 特征工程

### 2.1 特征整理流程

`src/data_preparation.py` 将多份 Excel 数据合并，并输出 `PreparedDatasets`：

1. **时间预处理**：统一转为 UTC，并生成 `time_to_close_hours/days` 等派生字段。
2. **文本统计**：对标题与正文计算词数、字符数、可读性指标，并构造布尔关键词特征。
3. **评论信息**：计算首条评论时间（Time-to-First Response）等互动特征。
4. **代码规模与结构**：整合 `lines_added/deleted`、`files_changed`、`directories`、`language_types` 等代码变更指标。
5. **作者/项目/评审画像**：将 `author_*`、`project_*`、`reviewer_*` 特征加前缀整合；引入作者历史 PR 数 `prev_prs`。
6. **数值清洗与缺失处理**：
   - 统一将布尔类型转为 0/1。
   - 删除重复或泄漏危险字段（详见 `DROP_FEATURE_COLUMNS` 与 `LEAKY_FEATURE_CANDIDATES`）。
   - 统一替换 inf/-inf 为 NaN。

### 2.2 关键特征列表（节选）

| 类别 | 代表特征 | 说明 |
| ---- | -------- | ---- |
| 文本与关键词 | `title_words`、`body_words`、`has_test`/`has_bug` 等 | 通过简单分词和字符串匹配获取文本长度与关键词布尔特征 |
| 代码规模 | `lines_added`、`lines_deleted`、`files_changed`、`segs_changed` | 量化改动范围，反映评审工作量 |
| 结构复杂度 | `directories`、`language_types`、`file_types` | 统计涉及目录层级与语言、文件类型多样性 |
| 开发者经验 | `prev_prs`、`author_*` | 作者历史与画像特征，衡量贡献者熟悉度 |
| 项目背景 | `project_*` | 项目级活动指标，例如活跃贡献者数量、评审轮次等 |
| 评审网络 | `reviewer_*`、`reviewer_count` 等 | 描述评审团队规模与角色 |
| 互动行为 | `comments`、`review_comments`、`time_to_first_response_hours` | 反映 PR 互动强度与响应速度 |

### 2.3 缺失值与异常处理

- 在模型训练前，通过 `fill_missing_within_features` 和 `fill_missing_train_test` 使用训练集的中位数填补数值缺失；当训练集合列全为 NaN 时回退为 0。
- 字符串类字段在统计前 `fillna('')`，确保计数函数正常运行。
- 通过 `replace([np.inf, -np.inf], np.nan)` 去除极端值干扰。

## 3. 建模与实验流程

### 3.1 流水线概览

`src/pipeline.py` 将数据工程与建模串联：

1. 调用 `build_feature_table` 生成聚合特征。
2. 基于固定时间阈值切分训练/测试集。
3. 进行数值特征填补并保持列顺序一致。
4. 分别训练回归与分类基线模型，输出指标表（CSV）与图像（PNG）。
5. 自动构建特征组（文本、结构、代码 churn、作者画像、项目画像、评审网络），执行特征消融实验并生成对比图。

### 3.2 回归模型（任务一）

模型定义见 `src/modeling.py`：

- `LinearRegression` + 标准化：提供线性基线。
- `RandomForestRegressor`：`n_estimators=300`、`min_samples_leaf=2`、`random_state=42`，能捕获非线性关系。
- `GradientBoostingRegressor` 与 `HistGradientBoostingRegressor`：对复杂特征进行逐步拟合。
- `TransformedTargetRegressor` 包裹的梯度提升模型：对目标应用 `log1p`/`expm1` 变换缓解长尾分布。

### 3.3 分类模型（任务二）

- `LogisticRegression`：`max_iter=1000`、`solver='lbfgs'`，验证线性可分性。
- `RandomForestClassifier`：`n_estimators=400`、`min_samples_leaf=2`，强调稳健性。
- `GradientBoostingClassifier`：关注精细划分边界。
- 所有线性模型均结合 `ColumnTransformer` 对数值特征进行标准化。

### 3.4 评估指标与产出

- 回归：MAE、MSE、RMSE、R²。
- 分类：Accuracy、Precision Macro、Recall Macro、F1 Macro。
- 所有结果写入 `outputs/`：
  - `regression_metrics.csv` / `classification_metrics.csv`
  - `regression_feature_ablation.csv` / `classification_feature_ablation.csv`
  - `figures/` 中的 PNG 图（指标对比、特征消融）。

## 4. 实验结果与分析

### 4.1 时间预测（单位：小时）

| 模型 | MAE | RMSE | R² |
| --- | --- | --- | --- |
| LinearRegression | 1325.82 | 1797.84 | -2.35 |
| RandomForest | 542.29 | 1142.75 | -0.35 |
| GradientBoosting | 575.84 | 1436.11 | -1.14 |
| HistGradientBoosting | 471.69 | 1089.13 | -0.23 |
| GradientBoosting + log1p(y) | 212.92 | 976.13 | 0.01 |

- 目标分布长尾导致线性与树模型在原尺度表现欠佳，R² 为负。
- 对 `time_to_close_hours` 进行 `log1p` 变换显著改善表现，MAE 降至约 212 小时（≈ 8.8 天），说明需要关注“极慢处理”的长尾样本。
- 消融结果显示作者画像与项目画像特征最关键，移除后 MAE 明显上升；文本与结构特征的影响较小。

### 4.2 合并预测

| 模型 | Accuracy | Precision Macro | Recall Macro | F1 Macro |
| --- | --- | --- | --- | --- |
| LogisticRegression | 0.878 | 0.846 | 0.676 | 0.718 |
| RandomForest | 0.888 | 0.867 | 0.701 | 0.747 |
| GradientBoosting | 0.883 | 0.804 | 0.752 | 0.773 |

- 梯度提升分类器在 Macro-F1 上表现最佳（0.77），兼顾正负样本的召回率。
- 随机森林提供最高准确率（0.889），但对少数类的召回略低。
- 特征消融表明作者相关特征对预测合并概率最为敏感，移除后 Macro-F1 降至 ~0.62；文本与结构特征影响相对有限。

### 4.3 图表与可视化

- 指标柱状图与消融对比图位于 `outputs/figures/`。

## 5. 结论与建议

- **作者与项目画像是核心信号**：无论回归还是分类，相关特征的缺失都会显著影响模型性能，建议项目维护者维护完整的贡献者画像数据。
- **处理时长具有明显长尾**：建议在运维中重点关注极端慢处理的 PR，可结合 `log1p` 变换或分位数回归进行建模。
- **文本特征贡献有限但可扩展**：当前仅采用简单关键词与长度统计，可考虑引入预训练文本嵌入以提升分类性能。

## 6. 项目结构与使用指南

```
.
├── src/
│   ├── data_preparation.py   # 数据读取与特征构建
│   ├── modeling.py           # 模型定义与评估函数
│   └── pipeline.py           # 端到端实验流程
├── GithubPR/                # GitHub PR 数据爬虫脚本
├── new_data_project/        # 多仓库数据处理与扩展实验
├── data/                    # 原始 Excel 数据
├── outputs/                 # 实验结果（运行后生成）
├── requirements.txt
└── README.md
```

### 6.1 环境准备

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Bash: source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 运行基线实验

```bash
python -m src.pipeline
```

运行结束后可在 `outputs/` 查看特征表、指标 CSV 与图像文件。

### 6.3 多仓库流程

进入 `new_data_project/` 目录后：

```bash
make integrate          # 整合多仓库数据
make run                # 同时执行回归与分类
make run MODE=regressor # 仅执行回归
make run MODE=classifier# 仅执行分类
```

特征消融分组可在 `new_data_project/src/config` 中配置。

## 7 其他

### 7.1 各人贡献

231250119 谢卓凡：队长，负责拥有数据后的代码特征提取以及训练等，代码整合，文档编写

231250051 刘佳昱：队员，负责爬取新项目数据，以及特征工程的内容。

### 7.2 代码仓库

代码开源在GitHub上，地址[wanyuanshenwanda/MLhw](https://github.com/wanyuanshenwanda/MLhw)，分支hw1

