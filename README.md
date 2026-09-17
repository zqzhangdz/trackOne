# 储层参数（POR/PERM/SW）智能预测

## 一、算法名称

基于默认/储层分段判别 + 梯度提升回归的测井储层参数预测模型。

## 二、方法简介

数据存在明显的“非储层默认段”与“储层真实段”二分结构：

- POR 约 66.5% 深度点为默认值 0.1；
- PERM 约 67.5% 深度点为默认值 0.01；
- SW 约 71.7% 深度点为默认值 99.9。

因此采用**逐目标两阶段**建模：

1. 第一阶段用 `HistGradientBoostingClassifier` 判别某深度点是否为默认段；
2. 第二阶段用 `HistGradientBoostingRegressor` 只对真实段回归；
3. 推理时若判为默认段则直接输出默认常数，否则输出回归值。

特征工程：剔除缺失率约 99% 的 `CASE`，保留 13 条输入曲线；将
`-99999/-9999/-2.5346e37` 等哨兵值统一为缺失，`CNL` 做物理截断，并加入
`logRT`、`logRXO`、`RT/RXO` 派生特征。`PERM` 在 log10 空间回归。

## 三、运行环境

- Python 3.10+
- 依赖见 `requirements.txt`

## 四、运行命令

预测（生成结果 JSON）：

```bash
python predict.py --data_dir ./data --output ./result.json
```

可选参数：

```bash
python predict.py --data_dir ./data --output ./result.json --model_dir ./models --config ./config.yaml
```

训练（复现模型）：

```bash
python train.py --data_dir ./train --out_dir ./models --cv_folds 5
```

## 五、文件说明

- `predict.py`：主预测脚本，完成数据读取、预处理、模型加载、推理与结果生成。
- `train.py`：训练脚本，按井 GroupKFold 交叉验证 + 阈值寻优，保存模型。
- `config.yaml`：模型路径与参数配置。
- `src/model.py`：两阶段模型定义与推理逻辑。
- `src/feature.py`：数据读取、哨兵值处理、特征工程。
- `src/utils.py`：配置常量、评价指标与通用工具。
- `models/model.pkl`：训练好的模型权重。
- `models/scaler.pkl`：特征列与预处理元信息。
- `examples/result_example.json`：结果输出样例。

## 六、评价指标

引入 `ε = 1e-5`，逐深度点计算截断准确率后取平均：

- `Acc_POR = mean(max(0, 1 - |ŷ-y| / (0.08·(|y|+ε))))`
- `Acc_PERM = mean(max(0, 1 - |log10(max(ŷ,ε))-log10(max(y,ε))| / (0.08·(|log10(max(y,ε))|+ε))))`
- `Acc_SW = mean(max(0, 1 - |ŷ-y| / (0.05·(|y|+ε))))`
- `Total = (0.3·Acc_POR + 0.35·Acc_PERM + 0.35·Acc_SW) × 100`

