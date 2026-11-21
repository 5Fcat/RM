"""换模型消融（`claude/正式实验计划.md` 变更记录第 23 条；只描述，结果出来之前写定）。

斜坡 5°（`slope_5`）、种子 500–509：把 B6（GA + 地形版 LLM 变异 + ② LLM Ev）里的 LLM 从 Haiku 5.5 整个换成别的模型
（变异与 ② 一起换；第 0–4 代的 B5 前缀也用新模型重跑），与已有的 B1（普通 GA）、Haiku 的 B6 配对比较。

用法：
  python -m hyblock.model_ablation_analysis --base <mac_runs/terrain_ext/formal/slope_5> \
      --root <mac_runs/model_ablation/slope_5> --tags sonnet [deepseek …] --out <目录> [--seeds 500-509]
  目录：<root>/ga_llmtyped_sel_llmEv_<tag>_s<种子>；B1 = <base>/ga_s<种子>，Haiku B6 = <base>/ga_llmtyped_sel_llmEv_s<种子>
  tag = haiku 时指向 Haiku 的 B6 本身（自检：B6′ − B6 = 0，B6′ − B1 = 地形扩展报告里的斜坡 +84.3）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hyblock.formal_analysis import boot_ci, run_metrics


def rng(t):
    a, b = t.split("-")
    return list(range(int(a), int(b) + 1))


def runs(base: Path, root: Path, tag: str, seeds) -> tuple[dict, list]:
    D, miss = {}, []
    for s in seeds:
        dirs = {"B1": base / f"ga_s{s}", "B6": base / f"ga_llmtyped_sel_llmEv_s{s}",
                "B6x": (base / f"ga_llmtyped_sel_llmEv_s{s}") if tag == "haiku"
                else root / f"ga_llmtyped_sel_llmEv_{tag}_s{s}"}
        for g, d in dirs.items():
            if (d / "DONE").exists():
                m = run_metrics(d)
                import csv
                g0 = [float(r["fitness"]) for r in csv.DictReader(open(d / "evals.csv", encoding="utf-8"))
                      if r["status"] == "mpc" and r["gen"] == "0"]
                m["g0"] = float(np.median(g0)) if g0 else float("nan")
                D[(g, s)] = m
            elif g == "B6x":
                miss.append(f"种子 {s}" + ("（没有 DONE）" if d.exists() else "（未开始）"))
    return D, miss


def _row(name, d):
    if len(d) < 2:
        return f"| {name} | {len(d)} | — | — | — | " + " ".join(f"{v:+.0f}" for v in d) + " |", None
    lo, hi = boot_ci(d)
    return (f"| {name} | {len(d)} | {np.mean(d):+.1f} | [{lo:+.1f}, {hi:+.1f}] | {sum(v > 0 for v in d)} / {len(d)} | "
            + " ".join(f"{v:+.0f}" for v in d) + " |"), (float(np.mean(d)), lo, hi)


def _pair(D, x, y, key, seeds):
    return [D[(x, s)][key] - D[(y, s)][key] for s in seeds if (x, s) in D and (y, s) in D]


def _mean(D, g, key, seeds):
    v = [D[(g, s)].get(key) for s in seeds if (g, s) in D and D[(g, s)].get(key) is not None]
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def analyse(base: Path, root: Path, tags, seeds) -> str:
    L = ["# 换模型消融：斜坡 5°（只描述）", "",
         "登记：`claude/正式实验计划.md` 变更记录第 23 条。脚本 `hyblock/model_ablation_analysis.py`（结果出来之前写定）。",
         "B1 = 普通 GA；B6 = GA + 地形版 LLM 变异 + ② LLM Ev（Haiku 5.5）；B6′ = 同 B6，LLM 换成该模型（变异与 ② 一起换）。", ""]
    for tag in tags:
        D, miss = runs(base, root, tag, seeds)
        ss = [s for s in seeds if ("B6x", s) in D]
        L += [f"## {tag}", ""]
        if miss:
            L += ["未完成：" + "、".join(miss), ""]
        if not ss:
            continue
        model = None
        import json
        d0 = (base / f"ga_llmtyped_sel_llmEv_s{ss[0]}") if tag == "haiku" else root / f"ga_llmtyped_sel_llmEv_{tag}_s{ss[0]}"
        try:
            model = json.loads((d0 / "run_meta.json").read_text(encoding="utf-8"))["args"].get("model")
        except Exception:
            pass
        L += [f"模型：`{model}`；完成的种子 {len(ss)} 个", "",
              "| 组 | 运行 | 最终最好 | 后期中位数 | 最好曲线均值 | 第 0 代中位数 | LLM 变异第 1 次通过 | ② 随机名额 Spearman | LLM 费用 / 运行 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for g, name in (("B1", "B1 GA"), ("B6", "B6 Haiku"), ("B6x", f"B6′ {tag}")):
            if any((g, s) in D for s in ss):
                ok = _mean(D, g, "mut_ok_first", ss)
                rho = _mean(D, g, "rand_rho", ss)
                usd = _mean(D, g, "usd", ss)
                L.append(f"| {name} | {sum((g, s) in D for s in ss)} | {_mean(D, g, 'best', ss):.0f} | {_mean(D, g, 'med', ss):.1f} | "
                         f"{_mean(D, g, 'auc', ss):.0f} | {_mean(D, g, 'g0', ss):.1f} | "
                         + ("—" if ok != ok else f"{ok:.2f}") + " | " + ("—" if rho != rho else f"{rho:+.2f}") + " | "
                         + ("—" if g == "B1" or usd != usd else f"${usd:.2f}") + " |")
        L += ["", "| 比较 | 种子 | 均值 | bootstrap 95% | 为正 | 各种子 |", "|---|---|---|---|---|---|"]
        res = {}
        for (x, y, key, desc) in (("B6x", "B6", "med", "B6′ − B6 后期中位数"), ("B6x", "B6", "best", "B6′ − B6 最终最好"),
                                  ("B6x", "B6", "auc", "B6′ − B6 最好曲线均值"), ("B6x", "B1", "med", "B6′ − B1 后期中位数"),
                                  ("B6x", "B1", "best", "B6′ − B1 最终最好"), ("B6", "B1", "med", "（参照）B6 − B1 后期中位数")):
            row, r = _row(desc, _pair(D, x, y, key, ss))
            L.append(row)
            res[(x, y, key)] = r
        g0d = [D[("B6x", s)]["g0"] - D[("B1", s)]["g0"] for s in ss if ("B1", s) in D]
        L += ["", "检查：第 0 代中位数与 B1 相同的种子 "
              f"{sum(abs(v) < 1e-9 for v in g0d)} / {len(g0d)}（第 0 代是同一种子的随机设计，应全部相同）", ""]
        r1, r2 = res[("B6x", "B6", "med")], res[("B6x", "B1", "med")]
        if r1 is not None and r2 is not None:
            m1, lo1, hi1 = r1
            m2, lo2, hi2 = r2
            a = ("高于 Haiku" if lo1 > 0 else "低于 Haiku" if hi1 < 0 else "与 Haiku 分不开")
            b = ("仍高于 GA" if lo2 > 0 else "低于 GA" if hi2 < 0 else "与 GA 分不开")
            L += [f"事先定的读法（后期中位数，bootstrap 95% 区间是否含 0）：换成 {tag} 后整体质量 **{a}**；**{b}**"
                  + ("" if len(ss) == len(seeds) else f"（只有 {len(ss)} / {len(seeds)} 个种子，未完成）"), ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--seeds", default="500-509")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    text = analyse(Path(a.base), Path(a.root), a.tags, rng(a.seeds))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
