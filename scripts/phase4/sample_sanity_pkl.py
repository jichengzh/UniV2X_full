"""阶段 4.4 — 从 cooperative train pkl 抽前 N 个 sample 做 sanity 训练子集.

理由: cooperative train 只 365 sample / 12 clips,全集 1 epoch 也快.
但首跑用前 50 sample(覆盖 2 个 clip)做最快 sanity,~5-10 min 看 loss 是否下降.
"""

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/home/jichengzhi/UniV2X/data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_train.pkl")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--dst", default="/home/jichengzhi/UniV2X/data/phase4/spd_train_sanity_50.pkl")
    args = parser.parse_args()

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)

    with open(args.src, "rb") as f:
        d = pickle.load(f)

    print(f"[src] {args.src}")
    print(f"  total samples: {len(d['infos'])}")
    print(f"  unique clips: {len(set(s['scene_token'] for s in d['infos']))}")

    # 抽前 N 个,但确保 prev/next 链合理(只保留 prev/next 都在子集内的 sample)
    subset = d["infos"][: args.n]
    subset_tokens = set(s["token"] for s in subset)
    # 修正 prev/next:链外的设 None
    for s in subset:
        if s.get("prev") and s["prev"] not in subset_tokens:
            s["prev"] = None
        if s.get("next") and s["next"] not in subset_tokens:
            s["next"] = None

    new_d = {"infos": subset, "metadata": d.get("metadata", {})}
    with open(args.dst, "wb") as f:
        pickle.dump(new_d, f)

    print(f"\n[dst] {args.dst}")
    print(f"  samples: {len(subset)}")
    print(f"  unique clips: {len(set(s['scene_token'] for s in subset))}")
    size_mb = Path(args.dst).stat().st_size / 1e6
    print(f"  size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
