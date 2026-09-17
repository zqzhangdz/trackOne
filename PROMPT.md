# 赛道一：储层参数（POR / PERM / SW）智能预测 — 完整 Prompt

---

## 一、任务定义

**输入**：14 条测井曲线（逐深度点，采样间隔 0.1m），一口井一个 `.txt` 文件。

**输出**：逐深度点回归三个储层参数：

| 目标 | 含义 | 默认常数值 | 默认段占比 |
|------|------|-----------|-----------|
| POR  | 孔隙度 | 0.1 | 66.5% |
| PERM | 渗透率 | 0.01 | 67.5% |
| SW   | 含水饱和度 | 99.9 | 71.7% |

数据存在明显的"非储层默认段"（三个目标同时退化为默认常数）与"储层真实段"二分结构。默认段预测准确即满分，瓶颈在真实段回归。

**数据规模**：训练 80 井 / ~730K 深度点；测试 10 井 / ~96K 深度点。

---

## 二、数据格式

### 训练文件（含标签，17 列）
```
DEPTH,GR,PE,SP,CAL,AC,DEN,CNL,RXO,RT,DEVI,AZIM,BIT,CASE,POR,PERM,SW
```
- 前 2 行为头部注释（列名 + 单位），实际数据从第 3 行开始。
- CASE 列缺失率 ~99%，直接剔除。
- 哨兵值：`-99999`、`-9999`、`-2.534573822542048e37`，均视为缺失。
- CNL 物理范围 [0, 100]%，超出视为测井误差需截断。

### 测试文件（无标签，14 列）
```
DEPTH,GR,PE,SP,CAL,AC,DEN,CNL,RXO,RT,DEVI,AZIM,BIT,CASE
```

---

## 三、评价指标

引入 ε = 1e-5，逐深度点计算截断准确率（各深度点独立、等权）：

```
Acc_POR  = mean( max(0, 1 − |ŷ − y| / (0.08 · (|y| + ε)) ) )
Acc_PERM = mean( max(0, 1 − |log10(max(ŷ,ε)) − log10(max(y,ε))| / (0.08 · (|log10(max(y,ε))| + ε)) ) )
Acc_SW   = mean( max(0, 1 − |ŷ − y| / (0.05 · (|y| + ε)) ) )
```

**总分**：`Total = (0.3 × Acc_POR + 0.35 × Acc_PERM + 0.35 × Acc_SW) × 100`

- POR 容忍 8% 相对误差（普通空间）；PERM 容忍 8% 相对误差（log10 空间）；SW 容忍 5% 相对误差。
- 默认常数值预测准确的点直接得满分（= 1.0 / 点），所以默认段分类正确即可拿到很高的基线分。

---

## 四、提交规范

需要提交两个独立 zip 文件：

### 4.1 `result.zip` — 预测结果
```json
{
  "modelId": "",
  "modelName": "模型名称",
  "version": "1.0",
  "resultData": [
    {
      "logId": "d122f5424b9e4984a8f701d16c2d5b58",
      "predictions": [
        { "depth": 1117.1, "POR": 21.7214, "PERM": 16.624, "SW": 79.0923 },
        { "depth": 1117.2, "POR": 21.2998, "PERM": 16.4908, "SW": 81.0157 }
      ]
    }
  ]
}
```
- logId = 文件名（不含 `.txt`）
- predictions 按 depth 升序
- 数值保留合理精度（建议 4 位小数）

### 4.2 `submission_code.zip` — 可复现代码（严格目录结构）
```
submission_code.zip
├── README.md          # 算法说明
├── predict.py         # 主预测入口
├── requirements.txt   # Python 依赖
├── config.yaml        # 配置文件
├── train.py           # 训练脚本（可选但推荐）
├── src/
│   ├── model.py       # 模型定义
│   ├── feature.py     # 特征工程
│   └── utils.py       # 工具函数
├── models/
│   ├── model.pkl      # 训练好的模型
│   └── scaler.pkl     # 预处理元信息
└── examples/
    └── result_example.json
```

**复现命令**：
```bash
pip install -r requirements.txt
python predict.py --data_dir ./data --output ./result.json
```

---

## 五、核心数据特征（决定建模策略）

1. **默认段占主导**：POR 66.5% / PERM 67.5% / SW 71.7% 的点为默认常数。纯默认输出可得 **69.20 分**。
2. **真实段三目标高度重叠**（Jaccard > 0.85），同井内沿深度高度平滑（POR lag-1 自相关 0.967，前一点持久性准确率 65%）。
3. **特征-目标相关性**：POR ↔ DEN(-0.72) / AC(0.69)；SW ↔ GR(-0.27) / RXO(-0.26)；POR ↑ → PERM ↑（相关 0.64）。
4. **真实段回归极难**：当前最优树模型只有 POR 17% / PERM 7% / SW 15% 的准确率。

---

## 六、当前成果（基线）

### 6.1 两阶段树模型（HistGradientBoosting）— 当前最优

**思路**：逐目标两阶段：
- 阶段 1：`HistGradientBoostingClassifier` 判别默认/真实
- 阶段 2：`HistGradientBoostingRegressor` 仅对真实段回归
- 推理时：分类概率 ≥ 阈值 → 默认常数，否则 → 回归值

**特征工程**：
- 剔除 CASE（缺失率 ~99%），保留 13 条原始曲线 + 3 个派生特征（logRT、logRXO、RT/RXO）= 16 维
- 哨兵值 → NaN，CNL 截断 [0, 100]
- PERM 在 log10 空间回归（匹配评价指标）

**参数**：`learning_rate=0.1/0.08`, `max_iter=300/400`, `max_leaf_nodes=31`, `min_samples_leaf=50`

**5 折 GroupKFold CV 结果**：**70.63 ± 1.43 分**
- 纯默认输出上界参考：69.20
- 真实段准确率：POR 17%、PERM 7%、SW 15%

### 6.2 双向 GRU 序列模型 — 未超越树基线
- 16 维特征 → 双向 GRU(128, 2 layers) → 多任务头（分类 + 回归）
- 按井采样 512 长度窗口
- 验证集约 62-64 分

### 6.3 自回归因果 GRU — 未超越树基线
- 输入 = 测井特征 + 前一点目标反馈信号
- 逐点推理，利用沿深度平滑先验
- 得分同样 ~62-64 分

### 6.4 已尝试但无效的方法
- 物理公式约束（DEN→POR、AC→POR）
- 预测后平滑
- 级联模型
- Total Variation 去噪

---

## 七、优化方向（按优先级排列）

### 方向 1：自回归 + Scheduled Sampling ★★★
核心洞察：前一点目标值持久性准确率 **65%**，而模型回归仅 7-17%。说明"相邻点值相近"这个先验比模型本身强得多。

方案：
- 训练时用 teacher forcing（输入真实前一点值），但概率用退火 Scheduled Sampling 逐步切换到模型自回归
- 推理时全自回归，前一点预测值反馈到当前点
- 这能直接利用 0.967 的深度自相关结构

### 方向 2：1D-CNN / TCN 替代 GRU ★★
固定感受野的卷积可能比 GRU 更适合这种局部平滑信号。TCN（Temporal Convolutional Network）带 dilation 可控制感受野大小。

### 方向 3：真实段样本加权 / Metric-Aware 损失 ★★
- 对真实段样本加权（如按默认/真实比例取反权重）
- 设计与评价指标一致的损失函数（截断相对误差的平滑近似）
- 对 PERM 的 log10 空间加权

### 方向 4：目标间物理正则 ★
POR ↑ → PERM ↑（相关 0.64），可在多任务学习中添加物理一致性正则项。

### 方向 5：更深树集成 + 超参搜索 ★
- CatBoost / LightGBM 替代 HistGradientBoosting
- 更细粒度的超参搜索（learning_rate、max_iter、max_depth、正则化）
- 每个目标独立调参而非共享参数

### 方向 6：分类-回归联合优化
当前两个阶段独立训练。可考虑：
- 将分类器的预测概率作为回归器的输入特征
- 或 end-to-end 多任务学习（共享 backbone，分类+回归联合训练）

### 方向 7：集成
- 树模型 + 序列模型的加权集成
- 不同随机种子的树模型 bagging

---

## 八、关键代码路径

当前项目位于 `/Users/yourmac/Downloads/THUNDER/赛道一/`：

| 文件 | 作用 |
|------|------|
| `src/model.py` | TwoStageModel（HistGradientBoosting 分类+回归） |
| `src/feature.py` | 数据读取、哨兵处理、特征工程 |
| `src/utils.py` | 常量、评价指标、输出工具 |
| `train.py` | 5-fold GroupKFold CV 训练 |
| `predict.py` | 预测入口（`--data_dir --output`） |
| `solution/model.py` | 同上（另一份拷贝） |
| `solution/seq_model.py` | 双向 GRU 序列模型 |
| `solution/ar_model.py` | 自回归因果 GRU |
| `train_seq.py` | GRU 训练脚本 |
| `train_ar.py` | 自回归 GRU 训练脚本 |
| `solution/metric.py` | 评价指标 |
| `solution/config.py` | 常量配置 |
| `solution/data.py` | 数据读取 |

---

## 九、目标

从当前 **70.63 分** 优化到 **75 分以上**。关键是把真实段回归准确率从 7-17% 提升到 40%+。只要真实段不再"几乎乱猜"，跨过 75 分线是可行的。
