"""全局配置与常量."""

# 原始曲线列（测试集没有最后三列）
RAW_COLS = [
    "DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL",
    "RXO", "RT", "DEVI", "AZIM", "BIT", "CASE",
]
TRAIN_COLS = RAW_COLS + ["POR", "PERM", "SW"]

# 实际参与建模的特征（CASE 缺失率 ~99%，直接剔除）
FEATURES = [
    "DEPTH", "GR", "PE", "SP", "CAL", "AC", "DEN", "CNL",
    "RXO", "RT", "DEVI", "AZIM", "BIT",
]

TARGETS = ["POR", "PERM", "SW"]

# 每个目标在“非储层/默认段”的主导常数
DEFAULT_VALUES = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}

# 数据中出现的缺失/异常哨兵值
SENTINEL_VALUES = [-99999.0, -9999.0, -2.534573822542048e37]

# 官方指标的极小值
EPS = 1e-5

# CNL 物理合理截断（单位 %，超出即视为测井误差）
CNL_LOW, CNL_HIGH = 0.0, 100.0

# 输出 clip 范围
OUTPUT_CLIP = {
    "POR": (0.0, 50.0),
    "PERM": (EPS, 5000.0),
    "SW": (0.0, 100.0),
}

# 官方总分权重
WEIGHTS = {"POR": 0.3, "PERM": 0.35, "SW": 0.35}

# 各目标的相对误差容差（POR/PERM 8%，SW 5%）
TOLERANCE = {"POR": 0.08, "PERM": 0.08, "SW": 0.05}

