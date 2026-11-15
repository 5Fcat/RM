"""地形扩展的预先登记分析（claude/地形扩展计划.md 第 2、4 节；11-13 写定，数据出来之前）。

pilot：每种地形选定难度上的普通 GA（开发种子 0–2）。通过 = 三个种子的后期中位数均值 ≥ 第 0 代中位数均值的 3 倍，
       且后期中位数均值 ≥ 20。不通过 → 换容易一级重新试点（最多一次）；仍不通过 → 该地形不进正式实验。
formal：每种地形（种子 500–509）B6 − B1 的后期中位数配对差（均值、各种子、为正个数、配对 bootstrap 95%），
       另报最终最好、最好曲线均值、各组均值、形态（模块数）。跨地形汇总并列平地（formal_flat A5 − A1）与横梁
       （formal_beams B6 − B1，种子 100–109）。只描述，不做显著性检验。
       事先定的读法：全部条件（新地形 + 平地 + 横梁）里平均差 > 0 的个数 ≥ 总数 − 1 →"在多种地形上方向一致"；
       ≥ 总数的一半 →"大体一致"；否则 →"受地形影响"。（登记时写的"6 个条件里 ≥ 5 个"按条件总数推广，
       进入正式实验的地形少于 6 种时同样适用。）
指标定义与 hyblock/formal_analysis.py 相同（后期 = 第 5 代起）。只用有 DONE 的运行；配对只用两组都完成的种子。

用法：python -m hyblock.terrain_ext_analysis pilot --root <mac_runs/terrain_ext/pilot> --out <目录>
      python -m hyblock.terrain_ext_analysis formal --root <mac_runs/terrain_ext/formal> --flat <formal_flat>
          --beams <formal_beams> --out <目录>
      自检：formal --root 指向一个目录，里面的子目录 beams_s1/ 链接到 formal_beams，--seeds 100-109
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import numpy as np

from hyblock.followup_analysis import load, paired
from hyblock.formal_analysis import boot_ci

G = {"B1": "ga", "B6": "ga_llmtyped_sel_llmEv"}


def terrain_dirs(root: Path):
    """root 下每种地形一个子目录（里面有 ga_s<种子>）；analysis 等其他目录跳过。"""
    return sorted(p for p in root.iterdir() if p.is_dir() and any(q.name.startswith("ga_s") for q in p.iterdir()))


def rng(t):
    a, b = t.split("-")
    return range(int(a), int(b) + 1)


def modules_mean(d: Path) -> float:
    rows = [r for r in csv.DictReader(open(d / "evals.csv", encoding="utf-8")) if r["status"] == "mpc"
            and int(r["gen"]) >= 5 and r.get("modules") not in (None, "")]
    return float(np.mean([float(r["modules"]) for r in rows])) if rows else float("nan")


def pilot(root: Path, seeds) -> tuple[str, dict]:
    L = ["# 地形扩展：GA 试点（种子 " + ",".join(map(str, seeds)) + "）", "",
         "通过 = 后期中位数均值 ≥ 3 × 第 0 代中位数均值，且 ≥ 20", "",
         "| 地形 | 完成种子 | 第 0 代中位数（各种子） | 后期中位数（各种子） | 最终最好均值 | 均值比 | 判定 |",
         "|---|---|---|---|---|---|---|"]
    verdict = {}
    for td in terrain_dirs(root):
        D, _ = load(td, {"B1": "ga"}, seeds)
        ss = [s for s in seeds if ("B1", s) in D]
        if len(ss) < len(seeds):
            L.append(f"| {td.name} | {len(ss)} / {len(seeds)} | — | — | — | — | 未完成 |")
            verdict[td.name] = "incomplete"
            continue
        g0 = [D[("B1", s)]["g0"] for s in ss]
        med = [D[("B1", s)]["med"] for s in ss]
        best = np.mean([D[("B1", s)]["best"] for s in ss])
        ratio = np.mean(med) / np.mean(g0) if np.mean(g0) > 0 else float("inf")
        ok = np.mean(med) >= 3 * np.mean(g0) and np.mean(med) >= 20
        verdict[td.name] = "pass" if ok else "fail"
        L.append(f"| {td.name} | {len(ss)} | {np.mean(g0):.1f}（{' '.join(f'{v:.0f}' for v in g0)}） | "
                 f"{np.mean(med):.1f}（{' '.join(f'{v:.0f}' for v in med)}） | {best:.0f} | {ratio:.1f} | "
                 f"{'通过' if ok else '不通过'} |")
    return "\n".join(L) + "\n", verdict


def _row(name, d):
    if len(d) < 2:
        return f"| {name} | {len(d)} | — | — | — | — |", (np.mean(d) if d else float("nan"))
    lo, hi = boot_ci(d)
    return (f"| {name} | {len(d)} | {np.mean(d):+.1f} | [{lo:+.1f}, {hi:+.1f}] | {sum(v > 0 for v in d)} / {len(d)} | "
            + " ".join(f"{v:+.0f}" for v in d) + " |"), float(np.mean(d))


def formal(root: Path, flat: Path | None, beams: Path | None, seeds, ref_seeds) -> str:
    L = ["# 地形扩展：正式实验（只描述）", "", f"种子 {seeds[0]}–{seeds[-1]}；B1 普通 GA，B6 GA + 地形版 LLM 变异 + ② LLM Ev。", ""]
    summ = []
    hdr = ["| 比较 | 种子 | 均值 | bootstrap 95% | 为正 | 各种子 |", "|---|---|---|---|---|---|"]
    for td in terrain_dirs(root):
        D, miss = load(td, G, seeds)
        L += [f"## {td.name}", ""]
        if miss:
            L += ["未完成：" + "、".join(miss), ""]
        L += ["| 组 | 运行 | 最终最好 | 后期中位数 | 最好曲线均值 | 第 0 代中位数 | 后期模块数 |", "|---|---|---|---|---|---|---|"]
        for g, name in G.items():
            ss = [s for s in seeds if (g, s) in D]
            if ss:
                mm = [modules_mean(td / f"{name}_s{s}") for s in ss]
                L.append(f"| {g} | {len(ss)} | {np.mean([D[(g, s)]['best'] for s in ss]):.0f} | "
                         f"{np.mean([D[(g, s)]['med'] for s in ss]):.1f} | {np.mean([D[(g, s)]['auc'] for s in ss]):.0f} | "
                         f"{np.mean([D[(g, s)]['g0'] for s in ss]):.1f} | {np.nanmean(mm):.1f} |")
        L += [""] + hdr
        for key, desc in (("med", "B6 − B1 后期中位数"), ("best", "B6 − B1 最终最好"), ("auc", "B6 − B1 最好曲线均值")):
            _, d = paired(D, "B6", "B1", key, seeds)
            row, mean = _row(desc, d)
            L.append(row)
            if key == "med":
                summ.append((td.name, len(d), mean, sum(v > 0 for v in d)))
        L.append("")
    if flat is not None:
        D, _ = load(flat, {"A1": "ga", "A5": "ga_llmtyped_sel_llmEv"}, ref_seeds)
        _, d = paired(D, "A5", "A1", "med", ref_seeds)
        summ.append(("flat（formal_flat A5 − A1）", len(d), float(np.mean(d)) if d else float("nan"), sum(v > 0 for v in d)))
    if beams is not None:
        D, _ = load(beams, G, ref_seeds)
        _, d = paired(D, "B6", "B1", "med", ref_seeds)
        summ.append(("beams（formal_beams B6 − B1）", len(d), float(np.mean(d)) if d else float("nan"), sum(v > 0 for v in d)))
    L += ["## 跨地形汇总：后期中位数配对差（LLM 变异 + ② − GA）", "", "| 条件 | 种子 | 均值 | 为正 |", "|---|---|---|---|"]
    L += [f"| {n} | {k} | {m:+.1f} | {p} / {k} |" for n, k, m, p in summ]
    valid = [m for _, k, m, _ in summ if k >= 2]
    npos, ntot = sum(m > 0 for m in valid), len(valid)
    if ntot:
        reading = ("在多种地形上方向一致" if npos >= ntot - 1 else "大体一致" if npos >= ntot / 2 else "受地形影响")
        L += ["", f"事先定的读法：平均差 > 0 的条件 {npos} / {ntot} → **{reading}**"]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["pilot", "formal"])
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default=None)
    p.add_argument("--flat", default=None)
    p.add_argument("--beams", default=None)
    p.add_argument("--ref_seeds", default="100-109")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "pilot":
        text, verdict = pilot(Path(a.root), list(rng(a.seeds or "0-2")))
        import json
        (out / "pilot_verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    else:
        text = formal(Path(a.root), Path(a.flat) if a.flat else None, Path(a.beams) if a.beams else None,
                      list(rng(a.seeds or "500-509")), list(rng(a.ref_seeds)))
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
