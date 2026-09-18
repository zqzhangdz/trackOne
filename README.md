# 储层参数（POR/PERM/SW）智能预测

## 一、算法名称

默认/储层分段判别 + 梯度提升树与双向 TCN 级联自回归融合的测井储层参数预测模型。

## 二、方法简介

数据存在"非储层默认段"（POR=0.1 / PERM=0.01 / SW=99.9，占 66–72% 深度点）与
"储层真实段"二分结构；真实段三目标沿深度高度平滑（lag-1 自相关 0.967）。
据此采用三级结构：

1. **分段判别**：逐目标 `HistGradientBoostingClassifier` 判别默认/真实段；
   特征为 13 条曲线 + 派生电阻率特征 + 单井上下文特征（lag/lead ±1,2,3,5,10、
   窗口 5/11/21 均值/标准差、一阶差分）+ 测井物理先验（密度/声波/中子孔隙度）；
   真实段回归器施加 POR 对 DEN 单调减、对 AC 单调增约束。
2. **双向 TCN pass1**：非因果时序卷积（感受野 ≈25 m）在 log10 空间回归三目标，
   损失为指标对齐的截断相对误差（带内线性、带外截断）+ 默认段 BCE。
3. **融合与决策**：树回归与 TCN 回归在 log10 空间按 0.3/0.7 加权；分段概率 ≥ 阈值
   （验证集调优：POR 0.10–0.15、PERM 0.05、SW 0.05）输出默认常数，否则输出融合值。
   真实段回归采用 metric-aware 训练：样本权重 ∝ 指标容差宽度（POR/SW 为 |y|、
   PERM 为 |log10 y|），把模型容量集中在"能得分"的深度点。
   （可选 pass2：以 pass1 因果平滑预测为通道的级联自回归 TCN，`train.py --with_pass2 1`；
   验证集无增益，默认关闭。）

推理全程纯 numpy（TCN 前向手写导出），提交包不依赖 torch。

## 三、运行环境

- Python 3.10+；依赖见 `requirements.txt`（训练 TCN 阶段另需 torch，缺失时自动退化为树模型）

## 四、运行命令

预测（生成结果 JSON）：

```bash
python predict.py --data_dir ./data --output ./result.json
```

训练（复现模型；分阶段子进程执行以隔离 OpenMP 运行时）：

```bash
python train.py --data_dir ./train --out_dir ./models --cv_folds 5 --with_tcn 1
```

## 五、文件说明

- `predict.py`：主预测脚本（数据读取、预处理、模型加载、混合推理、结果生成）
- `train.py`：训练脚本（树 CV/全量 + 两阶段 TCN + 工件组装）
- `config.yaml`：模型路径、名称版本、特征开关
- `src/model.py`：两阶段树模型、混合推理、纯 numpy TCN 前向、torch TCN 定义（懒加载）
- `src/feature.py`：数据读取、哨兵值处理、点式/上下文/物理先验特征
- `src/utils.py`：配置常量、官方评价指标、通用工具
- `models/model.pkl`：树模型 + TCN 权重（可选 pass2）+ 融合权重 + 阈值
- `models/scaler.pkl`：特征列与 TCN 输入标准化统计
- `examples/result_example.json`：结果输出样例

## 六、评价指标

引入 `ε = 1e-5`，逐深度点计算截断准确率后取平均：

- `Acc_POR = mean(max(0, 1 - |ŷ-y| / (0.08·(|y|+ε))))`
- `Acc_PERM = mean(max(0, 1 - |log10(max(ŷ,ε))-log10(max(y,ε))| / (0.08·(|log10(max(y,ε))|+ε))))`
- `Acc_SW = mean(max(0, 1 - |ŷ-y| / (0.05·(|y|+ε))))`
- `Total = (0.3·Acc_POR + 0.35·Acc_PERM + 0.35·Acc_SW) × 100`
