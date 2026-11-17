"""H7：跨地形合并的确认性检验（claude/地形扩展计划.md 变更记录第 3 条；11-15 写定，种子 500–509 没有任何数据之前）。

H7：在冻结进入正式实验的新地形上（wj_tools/terrain_ext_frozen.txt 列出的全部地形；不含已用于 H5 / H6 的
    平地与横梁 s = 1.0），GA + 地形版 LLM 变异 + ② LLM Ev（B6）的后期中位数高于普通 GA（B1）。
做法：每种地形 t、每个种子 s：d(t,s) = B6 − B1 的后期中位数；按地形把差标准化 z(t,s) = d(t,s) / σ_t，
     σ_t = 该地形 B1 后期中位数在种子之间的 SD（只用 B1，不用 B6；避免得分高的地形主导合并）；
     每个种子对各地形取平均 m_s = mean_t z(t,s)（同一种子在各地形上的运行用同一批初始随机数，按种子合并处理相关）；
     **主检验**：m_s（10 个种子）的单样本 t 检验，双侧 α = 0.05；单一假设，不与 H1–H6 合并校正。
     同时报告 Wilcoxon、bootstrap 95%、为正的种子数；次要（只描述）：不标准化的原始差按种子平均、各地形的差。
成立 = p < 0.05 且均值 > 0。冻结的新地形少于 2 种 → 不检验 H7（照实报告）。
用法：python -m hyblock.h7_analysis --dirs <地形目录>... --out <目录> [--seeds 500-509]
      （地形目录 = mac_runs/terrain_ext/formal/<地形名>，里面是 ga_s<种子> / ga_llmtyped_sel_llmEv_s<种子>）
自检：--dirs formal_flat formal_beams --seeds 100-109（平地 A5 = ga_llmtyped_sel_llmEv、A1 = ga，目录名与 B6 / B1 相同）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hyblock.followup_analysis import load
from hyblock.formal_analysis import boot_ci

G = {"B1": "ga", "B6": "ga_llmtyped_sel_llmEv"}


def rng(t):
    a, b = t.split("-")
    return range(int(a), int(b) + 1)


def analyse(dirs: list[Path], seeds) -> str:
    from scipy.stats import ttest_1samp, wilcoxon
    L = ["# H7：跨地形合并的确认性检验", "", "登记：`claude/地形扩展计划.md` 变更记录第 3 条。脚本 `hyblock/h7_analysis.py`（数据出来之前写定）。", "",
         "## 各地形", "", "| 地形 | 种子 | B1 后期中位数均值 | σ_t（B1 的 SD） | B6 − B1 均值 | 标准化均值 | 为正 |", "|---|---|---|---|---|---|---|"]
    Z, R = {}, {}
    for d in dirs:
        D, miss = load(d, G, seeds)
        ss = [s for s in seeds if ("B1", s) in D and ("B6", s) in D]
        b1 = [D[("B1", s)]["med"] for s in seeds if ("B1", s) in D]
        if len(ss) < 2 or len(b1) < 2:
            L.append(f"| {d.name} | {len(ss)} | — | — | — | — | 种子不足 |")
            continue
        sig = float(np.std(b1, ddof=1)) or 1.0
        diff = {s: D[("B6", s)]["med"] - D[("B1", s)]["med"] for s in ss}
        for s, v in diff.items():
            Z.setdefault(s, []).append(v / sig)
            R.setdefault(s, []).append(v)
        L.append(f"| {d.name} | {len(ss)} | {np.mean(b1):.0f} | {sig:.1f} | {np.mean(list(diff.values())):+.1f} | "
                 f"{np.mean([v / sig for v in diff.values()]):+.2f} | {sum(v > 0 for v in diff.values())} / {len(ss)} |")
    n_terr = sum(1 for d in dirs)
    L += ["", "## H7（主检验）", ""]
    if n_terr < 2:
        L.append("冻结的新地形少于 2 种 → 按登记不检验 H7。")
        return "\n".join(L) + "\n"
    ss = sorted(Z)
    m = np.array([np.mean(Z[s]) for s in ss])
    raw = np.array([np.mean(R[s]) for s in ss])
    t = ttest_1samp(m, 0.0)
    try:
        wp = wilcoxon(m).pvalue
    except ValueError:
        wp = float("nan")
    lo, hi = boot_ci(m)
    full = all(len(Z[s]) == n_terr for s in ss) and len(ss) == len(list(seeds))
    verdict = "成立" if (t.pvalue < 0.05 and m.mean() > 0) else "不成立"
    if not full:
        verdict += "（有缺失的运行，还不是最终）"
    L += ["| 量 | 种子 | 均值 | bootstrap 95% | 单样本 t | p | Wilcoxon p | 为正 | 判定 |", "|---|---|---|---|---|---|---|---|---|",
          f"| 标准化差的种子平均 m_s | {len(m)} | {m.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {t.statistic:.2f} | {t.pvalue:.4f} | "
          f"{wp:.3f} | {int((m > 0).sum())} / {len(m)} | **{verdict}** |", "",
          "- 各种子 m_s：" + " ".join(f"{s}:{v:+.2f}" for s, v in zip(ss, m)),
          f"- 次要（只描述）：原始差的种子平均 {raw.mean():+.1f} [{boot_ci(raw)[0]:+.1f}, {boot_ci(raw)[1]:+.1f}]，{int((raw > 0).sum())} / {len(raw)} 为正",
          f"- 合并的地形（{n_terr} 种）：" + "、".join(d.name for d in dirs)]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--seeds", default="500-509")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    txt = analyse([Path(d) for d in a.dirs], list(rng(a.seeds)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
