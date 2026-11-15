"""计算量对等分析（第 15 周，追加 B；claude/正式实验计划.md 变更记录第 21 条）。只用已有数据，只描述。

问题：LLM 变异与 ② 多花了 API 费用与等待 LLM 的时间。如果把这部分时间给 GA 多做评估，GA 能不能追上？
数据：Mac 任务日志里每代的 "propose Ns, gen Ms"（gen = 该代总墙钟时间，含 propose；各代之和 = 运行的墙钟时间，
     已用 formal_flat/ga_llmtyped_s100 核对：8126 s vs run_meta → DONE 8125 s）、evals.csv 的 mpc_s / screen_s、
     run_metrics 的费用。配对组（--init_from，复制前 5 代）的成本 = 来源运行前 5 代的时间 + 本运行第 5 代起的时间。
口径：(1) 每运行的时间构成（MPC / 非 MPC = 等 LLM 与其他）与费用；(2) LLM 组 / GA 的总时间之比；
     (3) 相同总时间：两组都在 t* = min(两者总时间) 处截取，比较到 t* 为止的最好得分与"后期"（第 5 代起）评估的中位数。
     LLM 组更快（比值 ≤ 1）时，相同总时间的比较对 LLM 组不会比相同评估次数更不利；比 GA 慢 20% 以上时按第 21 条另行登记小实验。
用法：python -m hyblock.compute_parity --runs <mac_runs> --logs <logs> --out <目录>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

import numpy as np

from hyblock.formal_analysis import boot_ci, run_metrics

LOGS = ["formal_flat.out", "week13_mac1.out", "week14_mac1.out"]
EXPS = {   # 实验根目录 → (日志里的前缀, [(组, 目录名)], [(LLM 组, 对照 GA 组)])
    "formal_flat": ("", [("A1", "ga"), ("A2", "ga_sel_llmEv"), ("A3", "ga_sel_knnEv"), ("A4", "ga_llmtyped"),
                         ("A5", "ga_llmtyped_sel_llmEv"), ("A6", "ga_sel_llmEv_sonnet")],
                    [("A2", "A1"), ("A3", "A1"), ("A4", "A1"), ("A5", "A1"), ("A6", "A1")]),
    "replicate_flat": ("replicate_flat/", [("A4", "ga_llmtyped"), ("A5", "ga_llmtyped_sel_llmEv")], [("A5", "A4")]),
    "formal_beams": ("formal_beams/", [("B1", "ga"), ("B2", "ga_sel_llmEv"), ("B3", "ga_init_transfer"),
                                       ("B4", "ga_init_llmtransfer"), ("B5", "ga_llmtyped"), ("B6", "ga_llmtyped_sel_llmEv")],
                     [("B2", "B1"), ("B4", "B1"), ("B5", "B1"), ("B6", "B1")]),
    "sensitivity/p14": ("p14/", [("A1", "ga"), ("A4", "ga_llmtyped"), ("A5", "ga_llmtyped_sel_llmEv")], [("A4", "A1"), ("A5", "A1")]),
    "sensitivity/m15": ("m15/", [("A1", "ga"), ("A4", "ga_llmtyped"), ("A5", "ga_llmtyped_sel_llmEv")], [("A4", "A1"), ("A5", "A1")]),
    "sensitivity/v370": ("v370/", [("A1", "ga"), ("A4", "ga_llmtyped"), ("A5", "ga_llmtyped_sel_llmEv")], [("A4", "A1"), ("A5", "A1")]),
}
HEAD = re.compile(r"^=== (\S+) 种子 (\d+) ")
GEN = re.compile(r"gen (\d+):.*\| propose (\d+)s, gen (\d+)s")


def parse_logs(log_dir: Path) -> dict:
    """{(日志里的名称, 种子): {代: (propose_s, gen_s)}}；续跑时后出现的同一代覆盖前面的。"""
    T, cur = {}, None
    for name in LOGS:
        p = log_dir / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = HEAD.match(line)
            if m:
                cur = (m.group(1), int(m.group(2)))
                T.setdefault(cur, {})
                continue
            if line.startswith("=== "):
                cur = None
                continue
            g = GEN.search(line)
            if g and cur:
                T[cur][int(g.group(1))] = (int(g.group(2)), int(g.group(3)))
    return T


def timeline(d: Path, gens: dict, src_gens: dict | None, sel_from: int) -> dict:
    """每次 MPC 评估完成时的累计墙钟时间（复制来的前几代用来源运行的时间）。"""
    rows = [r for r in csv.DictReader(open(d / "evals.csv", encoding="utf-8"))]
    by = {}
    for r in rows:
        by.setdefault(int(r["gen"]), []).append(r)
    t, out, parts = 0.0, [], {"mpc": 0.0, "other": 0.0}
    for g in sorted(by):
        G = (src_gens if (src_gens is not None and g < sel_from) else gens).get(g)
        if G is None:
            return {}
        prop, wall = G
        t0 = t + prop
        mpc_g = 0.0
        for r in sorted(by[g], key=lambda r: int(r["idx"])):
            if r["status"] == "mpc":
                t0 += float(r["mpc_s"] or 0) + float(r["screen_s"] or 0)
                mpc_g += float(r["mpc_s"] or 0)
                out.append((t0, g, float(r["fitness"])))
        parts["mpc"] += mpc_g
        parts["other"] += wall - mpc_g
        t += wall
    return {"evals": out, "total": t, **parts}


def at_time(tl: dict, tstar: float) -> dict:
    E = [e for e in tl["evals"] if e[0] <= tstar + 1e-6]
    late = [f for (_, g, f) in E if g >= 5]
    return {"best": max((f for _, _, f in E), default=float("nan")), "med": st.median(late) if late else float("nan"),
            "n": len(E)}


def analyse(runs: Path, logs: Path) -> str:
    T = parse_logs(logs)
    L = ["# 计算量对等分析（追加 B，描述性）", "",
         "登记：`claude/正式实验计划.md` 变更记录第 21 条。脚本 `hyblock/compute_parity.py`。时间 = Mac 墙钟（单进程 12 线程）；"
         "配对组的成本含来源运行前 5 代。", ""]
    worst = []
    for exp, (prefix, groups, pairs) in EXPS.items():
        root = runs / exp
        if not root.exists():
            continue
        D = {}
        for g, name in groups:
            for d in sorted(root.glob(f"{name}_s*")):
                m = re.match(rf"^{re.escape(name)}_s(\d+)$", d.name)
                if not m or not (d / "DONE").exists():
                    continue
                s = int(m.group(1))
                a = json.loads((d / "run_meta.json").read_text(encoding="utf-8")).get("args") or {}
                src = None
                if a.get("init_from"):
                    src_name = Path(a["init_from"]).parent.name.rsplit("_s", 1)[0]
                    src = T.get((prefix + src_name, s))
                tl = timeline(d, T.get((prefix + name, s), {}), src, int(a.get("sel_from", 1)))
                if not tl:
                    continue
                D[(g, s)] = {"tl": tl, "m": run_metrics(d)}
        if not D:
            continue
        L += [f"## {exp}", "", "| 组 | 运行 | 总时间 h | MPC h | 非 MPC h（等 LLM 等） | 非 MPC 占比 | 费用 $ / 运行 |", "|---|---|---|---|---|---|---|"]
        for g, _ in groups:
            S = [D[(g, s)] for (gg, s) in D if gg == g]
            if not S:
                continue
            tot = np.mean([x["tl"]["total"] for x in S]) / 3600
            mp = np.mean([x["tl"]["mpc"] for x in S]) / 3600
            ot = np.mean([x["tl"]["other"] for x in S]) / 3600
            L.append(f"| {g} | {len(S)} | {tot:.2f} | {mp:.2f} | {ot:.2f} | {ot / tot:.0%} | {np.mean([x['m']['usd'] for x in S]):.2f} |")
        L += ["", "| 比较 | 种子 | 总时间均值之比（LLM 组 / 对照） | 相同评估次数：后期中位数差 | 相同总时间：后期中位数差 | 相同评估次数：最终最好差 | 相同总时间：最好差 |",
              "|---|---|---|---|---|---|---|"]
        for x, y in pairs:
            ss = sorted(s for (g, s) in D if g == x and (y, s) in D)
            if len(ss) < 2:
                continue
            ratio, dm_e, dm_t, db_e, db_t = [], [], [], [], []
            for s in ss:
                X, Y = D[(x, s)], D[(y, s)]
                ratio.append(X["tl"]["total"] / Y["tl"]["total"])
                dm_e.append(X["m"]["med"] - Y["m"]["med"])
                db_e.append(X["m"]["best"] - Y["m"]["best"])
                ts = min(X["tl"]["total"], Y["tl"]["total"])
                ax, ay = at_time(X["tl"], ts), at_time(Y["tl"], ts)
                dm_t.append(ax["med"] - ay["med"])
                db_t.append(ax["best"] - ay["best"])

            def c(v, nd=0):
                v = [t for t in v if t == t]                 # 相同总时间时，较慢的一组可能在 t* 前还没有后期评估（nan，去掉）
                if len(v) < 2:
                    return "—"
                lo, hi = boot_ci(v)
                return f"{np.mean(v):+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]（{sum(t > 0 for t in v)} / {len(v)}）"
            r = np.array(ratio)
            rm = np.mean([D[(x, s)]["tl"]["total"] for s in ss]) / np.mean([D[(y, s)]["tl"]["total"] for s in ss])
            worst.append((exp, x, y, float(rm)))
            L.append(f"| {x} − {y} | {len(ss)} | **{rm:.3f}**（各种子 {r.min():.2f}–{r.max():.2f}） | {c(dm_e)} | {c(dm_t)} | {c(db_e)} | {c(db_t)} |")
        L.append("")
    over = [w for w in worst if w[3] > 1.20]
    L += ["## 判定（第 21 条）", "",
          ("LLM 组每运行的总时间都没有比对照多 20% 以上 → **不需要追加\"GA 放宽预算\"的小实验**。" if not over else
           "以下比较的 LLM 组总时间比对照多 20% 以上 → 按第 21 条另行登记\"GA 按相同总时间放宽预算\"的小实验：" +
           "；".join(f"{e} {x} − {y}（×{v:.2f}）" for e, x, y, v in over)),
          "", "- 相同总时间：两组都截到 t* = min(两者总时间)；较快的一组用完 200 次评估，较慢的一组只算到 t* 为止的评估",
          "- 非 MPC 时间主要是等 LLM（变异的 propose 与 ② 的排序）；GA 的非 MPC 时间只有筛选与编译（几秒）",
          "- 续跑过的运行：中断前没写完的那一代的时间没有记进日志，这类运行的总时间略偏小"]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--logs", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    txt = analyse(Path(a.runs), Path(a.logs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
