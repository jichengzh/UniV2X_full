"""精度预测器接口 (Stage S3.7)

对应 v1.5 + Phase1_2 实施计划 §1.2 接口契约.

接口:
- AccuracyPredictor.predict(config) -> float
    返回 accuracy_diff_vs_fp32 (越小越好)
- AccuracyPredictor.predict_with_uncertainty(config) -> (mean, std)
    返回均值 + 不确定性 (用于贝叶斯搜索 / 早停)

实现:
- LGBPredictor: LightGBM booster 包装 (S3.2 已训出 v0)
- 后续可扩展 MLPPredictor / TransferPredictor (Stage 3 升级路径)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd

from framework.config_schema import Config
from framework.feature_encoder import (
    CATEGORICAL_COLS,
    encode_config,
    encode_configs,
)


class AccuracyPredictor(Protocol):
    """精度预测器接口 (Phase1_2 §1.2 契约)."""

    def predict(self, config: Config) -> float:
        """返回精度差预测 (accuracy_diff_vs_fp32, 越小越好)."""
        ...

    def predict_with_uncertainty(self, config: Config) -> tuple[float, float]:
        """返回 (mean, std). 没有不确定性估计的实现可以返回 (mean, 0.0)."""
        ...


class LGBPredictor:
    """LightGBM 实现.

    用法:
        p = LGBPredictor.from_files(
            model_path='models/lgb_predictor_v0.txt',
            meta_path='models/lgb_predictor_v0_meta.json',
        )
        diff = p.predict(some_config)
    """

    def __init__(
        self,
        booster: lgb.Booster,
        feature_names: list[str],
        meta: Optional[dict] = None,
    ) -> None:
        self.booster = booster
        self.feature_names = feature_names
        self.meta = meta or {}

    # ------------- 加载 / 保存 -------------

    @classmethod
    def from_files(
        cls,
        model_path: str | Path,
        meta_path: Optional[str | Path] = None,
    ) -> "LGBPredictor":
        booster = lgb.Booster(model_file=str(model_path))
        meta: dict = {}
        if meta_path is not None and Path(meta_path).exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
        return cls(booster, booster.feature_name(), meta)

    # ------------- 单点预测 -------------

    def _encode(self, config: Config) -> pd.DataFrame:
        """单个 Config → DataFrame (列顺序与训练时对齐)."""
        row = encode_config(config)
        df = pd.DataFrame([row])
        # 类别列转 category dtype (LightGBM 期望)
        for c in CATEGORICAL_COLS:
            if c in df.columns:
                df[c] = df[c].astype("category")
        # 列顺序对齐 booster.feature_name()
        for fn in self.feature_names:
            if fn not in df.columns:
                df[fn] = pd.NA
        return df[self.feature_names]

    def predict(self, config: Config) -> float:
        X = self._encode(config)
        y = self.booster.predict(X)
        return float(y[0])

    def predict_with_uncertainty(self, config: Config) -> tuple[float, float]:
        """LightGBM 单模型无原生不确定性估计,std=0.

        Stage 3.6 升级路径: quantile regression (训三个 booster) 或 ensemble.
        """
        return self.predict(config), 0.0

    # ------------- 批量预测 (搜索器调用) -------------

    def predict_batch(self, configs: list[Config]) -> np.ndarray:
        """批量预测 (NSGA-II 一代 ~100 候选用得上)."""
        X = encode_configs(configs)
        # 列顺序对齐
        for fn in self.feature_names:
            if fn not in X.columns:
                X[fn] = pd.NA
        X = X[self.feature_names]
        return self.booster.predict(X)

    # ------------- 元信息 -------------

    @property
    def is_trustworthy(self) -> bool:
        """根据训练时的 Spearman ρ 判断是否值得用.

        sanity 阶段 (~0.5) 仅作 demo;搜索器应在 ρ > 0.7 时才信任输出.
        """
        m = self.meta.get("metrics_loocv", {})
        rho = m.get("spearman_rho", 0.0)
        return rho >= 0.70

    def trust_level(self) -> str:
        m = self.meta.get("metrics_loocv", {})
        rho = m.get("spearman_rho", 0.0)
        if rho >= 0.85:
            return "high"
        if rho >= 0.70:
            return "medium"
        if rho >= 0.50:
            return "low (sanity only)"
        return "untrustworthy"


# 默认预测器加载器 — 给搜索器用
def load_default_predictor() -> LGBPredictor:
    """加载默认 v0 预测器."""
    root = Path(__file__).resolve().parents[1]
    return LGBPredictor.from_files(
        model_path=root / "models" / "lgb_predictor_v0.txt",
        meta_path=root / "models" / "lgb_predictor_v0_meta.json",
    )


__all__ = ["AccuracyPredictor", "LGBPredictor", "load_default_predictor"]
