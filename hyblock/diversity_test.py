"""H8：多样性的确认性检验（登记见 `claude/补充分析A_2026-11-21.md`；脚本在计算 confirm_beams / 地形扩展的多样性之前写定）。

主要指标（每个运行）= 后期种群（第 5 代起的 MPC 评估）里**不同身体结构的个数**（细签名：根是模块 / 分支件、身体里的
分支件数、各肢的模块数、脚数）。次要（只描述 + 同样的检验，不作判定）：粗身体结构（链状 / 分支件 + k 肢）的香农熵。
签名的定义与 11-12 的探索性分析（`from_mac/mac_runs/morphology/diversity.py`）完全相同。

  H8a：横梁 s = 1.0、H6 的种子 400–423：B6 − B1 < 0，24 对配对 t（双侧）
  H8b：地形扩展 3 种地形、种子 500–509：每个种子对 3 种地形取 B6 − B1 的平均，10 个种子单样本 t（双侧）
  H8a、H8b 两个 p 一起 Holm 校正；成立 = Holm 后 p < 0.05 且均值 < 0

用法：python -m hyblock.diversity_test --confirm <confirm_beams> --confirm_seeds 400-423 \
          --terrains <formal/ridged_s0.75> <formal/slope_5> <formal/beams_s1.1> --terrain_seeds 500-509 --out <目录>
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
from pathlib import Path

import numpy as np

from hyblock.design import build_robot
from hyblock.formal_analysis import boot_ci

CACHE: dict = {}
KEYS = {"fine": "后期种群不同身体结构数（主要）", "H": "粗结构熵 bit（次要）"}
B1, B6 = "ga", "ga_llmtyped_sel_llmEv"


def sig(text):
    if text in CACHE:
        return CACHE[text]
    try:
        _, ex = build_robot(text, None, max_modules=10 ** 6, max_hubs=10 ** 6)
    except Exception:
        CACHE[text] = None
        return None
    kids = collections.defaultdict(list)
    for i, n in enumerate(ex.nodes):
        if n.parent >= 0:
            kids[n.parent].append(i)
    body = {0} if ex.nodes[0].kind == "hub" else set()
    st = list(body)
    while st:
        i = st.pop()
        for c in kids[i]:
            if ex.nodes[c].kind == "hub":
                body.add(c)
                st.append(c)

    def size(i):
        return ex.nodes[i].n_par * (ex.nodes[i].kind == "module") + sum(size(c) for c in kids[i])
    limbs = tuple(sorted((size(c) for b in body for c in kids[b] if ex.nodes[c].kind == "module"), reverse=True))
    mods = ex.module_count()
    s = ("hub" if body else "chain", len(body), limbs if body else (mods,), len(ex.feet))
    vec = (mods, ex.hub_count(), len(limbs), len(ex.feet), sum(n.n_par == 2 for n in ex.nodes), sum(ex.parity))
    CACHE[text] = (s, vec)
    return CACHE[text]


def coarse(s):
    return s[0] if s[0] == "chain" else f"hub{len(s[2])}limb"


def entropy(c: collections.Counter) -> float:
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else float("nan")


def run_div(d: Path) -> dict | None:
    if not (d / "DONE").exists():
        return None
    with open(d / "evals.csv", encoding="utf-8") as f:
        R = [r for r in csv.DictReader(f) if r["status"] == "mpc"]
    late = [x for x in (sig(r["design"]) for r in R if int(r["gen"]) >= 5) if x]
    seen, top = set(), []
    for r in sorted(R, key=lambda r: -float(r["fitness"])):
        if r["design"] not in seen:
            seen.add(r["design"])
            top.append(sig(r["design"]))
            if len(top) == 16:
                break
    return dict(H=entropy(collections.Counter(coarse(x[0]) for x in late)), fine=len({x[0] for x in late}),
                top16=len({x[0] for x in top if x}), sd_mod=float(np.std([x[1][0] for x in late])),
                sd_limb=float(np.std([x[1][2] for x in late])))


def rng(t):
    a, b = t.split("-")
    return list(range(int(a), int(b) + 1))


def pairs(root: Path, seeds, b1=B1, b6=B6) -> dict:
    out = {}
    for s in seeds:
        x, y = run_div(root / f"{b1}_s{s}"), run_div(root / f"{b6}_s{s}")
        if x and y:
            out[s] = (x, y)
    return out


def _desc(P: dict) -> list[str]:
    L = ["| 指标 | B1 均值 | B6 均值 | B6 − B1 | bootstrap 95% | B6 更低的种子 |", "|---|---|---|---|---|---|"]
    for k, name in (("H", "粗结构熵 bit（次要）"), ("fine", "后期种群不同身体结构数（细；H8 的主要指标）"), ("top16", "最终前 16 不同身体结构数"),
                    ("sd_mod", "后期模块数 SD"), ("sd_limb", "后期肢数 SD")):
        d = [P[s][1][k] - P[s][0][k] for s in P]
        lo, hi = boot_ci(d) if len(d) >= 2 else (float("nan"), float("nan"))
        L.append(f"| {name} | {np.mean([P[s][0][k] for s in P]):.2f} | {np.mean([P[s][1][k] for s in P]):.2f} | "
                 f"{np.mean(d):+.2f} | [{lo:+.2f}, {hi:+.2f}] | {sum(v < 0 for v in d)} / {len(d)} |")
    return L


def _test(d) -> tuple[float, float, float, float, float]:
    from scipy.stats import ttest_1samp, wilcoxon
    d = np.asarray(d, float)
    t, p = ttest_1samp(d, 0.0)
    try:
        pw = wilcoxon(d).pvalue
    except ValueError:
        pw = float("nan")
    lo, hi = boot_ci(d)
    return float(np.mean(d)), lo, hi, float(t), float(p), float(pw)


def analyse(confirm: Path | None, cseeds, terrains: list[Path], tseeds, key="fine", label_a="H8a", label_b="H8b") -> str:
    L = [f"# H8：多样性的确认性检验（B6 − B1；指标 {KEYS[key]}）", "",
         "登记：`claude/补充分析A_2026-11-21.md`。脚本 `hyblock/diversity_test.py`（计算这些数据的多样性之前写定）。", ""]
    res = {}
    if confirm is not None:
        P = pairs(confirm, cseeds)
        L += [f"## {label_a}：{confirm.name}（{len(P)} 对）", ""] + _desc(P) + [""]
        res[label_a] = [P[s][1][key] - P[s][0][key] for s in sorted(P)]
    if terrains:
        per = collections.defaultdict(list)
        for td in terrains:
            P = pairs(td, tseeds)
            L += [f"## {td.name}（{len(P)} 对，描述）", ""] + _desc(P) + [""]
            for s in P:
                per[s].append(P[s][1][key] - P[s][0][key])
        full = sorted(s for s in per if len(per[s]) == len(terrains))
        res[label_b] = [float(np.mean(per[s])) for s in full]
        L += [f"{label_b} 用的种子（{len(terrains)} 种地形都齐）：{len(full)} 个；各种子平均差 "
              + " ".join(f"{s}:{np.mean(per[s]):+.2f}" for s in full), ""]
    from hyblock.followup_analysis import holm
    ps, rows = {}, []
    for k, d in res.items():
        if len(d) >= 3:
            m, lo, hi, t, p, pw = _test(d)
            ps[k] = p
            rows.append((k, len(d), m, lo, hi, t, p, pw, sum(v < 0 for v in d)))
    hp = holm(ps)
    L += ["## 检验", "", "| 假设 | n | 均值 | bootstrap 95% | t | p | Holm p | Wilcoxon p | B6 更低 | 判定 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, n, m, lo, hi, t, p, pw, neg in rows:
        ok = hp[k] < 0.05 and m < 0
        L.append(f"| {k} | {n} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {t:.2f} | {p:.4f} | {hp[k]:.4f} | {pw:.4f} | {neg} / {n} | "
                 + ("**成立**" if ok else "不成立") + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--confirm")
    p.add_argument("--confirm_seeds", default="400-423")
    p.add_argument("--terrains", nargs="*", default=[])
    p.add_argument("--terrain_seeds", default="500-509")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    args = (Path(a.confirm) if a.confirm else None, rng(a.confirm_seeds), [Path(t) for t in a.terrains], rng(a.terrain_seeds))
    text = analyse(*args, key="fine")
    sec = analyse(*args, key="H").split("## 检验", 1)[1]
    text += "\n## 次要：粗结构熵的同样检验（只描述，不作判定）" + sec
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
