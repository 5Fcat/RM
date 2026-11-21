"""换模型消融（正式实验计划变更记录第 23 条）的 API 费用：按已记录的 token 数估算，用于分批时的费用上限。

只数真正调用了 API 的行（cached 为真的是缓存 / 复制来的，不算）：
  - B5′ 前缀 ga_llmtyped_pre_<标签>_s<种子>/llm_mut.csv 的全部行
  - B6′ ga_llmtyped_sel_llmEv_<标签>_s<种子>/llm_mut.csv 第 5 代起的行（第 0–4 代是从前缀复制的）
  - B6′ 的 select.csv（② 只在第 5 代起调用）
费用 = 输入 token × 输入单价 + 输出 token × 输出单价（USD / 百万 token）。只是估算：没有算提示缓存等折扣，偏保守。

用法：python -m hyblock.ablation_cost --root <mac_runs/model_ablation/slope_5> --tag sonnet --model claude-sonnet-5-5 [--max 55]
  有 --max 且估算 ≥ max 时退出码 3（任务脚本据此不再开始新的运行）。
"""
import argparse
import csv
import sys
from pathlib import Path

# USD / 百万 token（输入, 输出）；claude/修论pipeline.md 第 5 节\"候选模型与价格\"（11-16 各家官网）
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}


def _tokens(path: Path, gen_min: int = 0):
    tin = tout = calls = 0
    if not path.exists():
        return tin, tout, calls
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                if int(r.get("gen", 0)) < gen_min:
                    continue
            except ValueError:
                continue
            if str(r.get("cached", "")).strip().lower() in ("true", "1"):
                continue
            if str(r.get("llm_in", "")).strip() == "":
                continue
            tin += int(float(r["llm_in"]))
            tout += int(float(r.get("llm_out") or 0))
            calls += 1
    return tin, tout, calls


def seed_tokens(root: Path, tag: str, seed: int):
    parts = [
        _tokens(root / f"ga_llmtyped_pre_{tag}_s{seed}" / "llm_mut.csv"),
        _tokens(root / f"ga_llmtyped_sel_llmEv_{tag}_s{seed}" / "llm_mut.csv", gen_min=5),
        _tokens(root / f"ga_llmtyped_sel_llmEv_{tag}_s{seed}" / "select.csv"),
    ]
    return tuple(sum(p[i] for p in parts) for i in range(3))


def cost(root: Path, tag: str, price, seeds=range(500, 510)):
    rows, total = [], 0.0
    for s in seeds:
        tin, tout, calls = seed_tokens(root, tag, s)
        if calls == 0:
            continue
        usd = tin * price[0] / 1e6 + tout * price[1] / 1e6
        rows.append((s, tin, tout, calls, usd))
        total += usd
    return total, rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--price", nargs=2, type=float, metavar=("IN", "OUT"), help="USD / 百万 token；不在 PRICES 里的模型必须给")
    p.add_argument("--max", type=float, default=None)
    a = p.parse_args(argv)
    price = tuple(a.price) if a.price else PRICES.get(a.model)
    if price is None:
        print(f"!!! 不知道 {a.model} 的单价，用 --price 给出", file=sys.stderr)
        return 2
    total, rows = cost(Path(a.root), a.tag, price)
    for s, tin, tout, calls, usd in rows:
        print(f"  {a.tag} 种子 {s}：{calls} 次调用，输入 {tin} / 输出 {tout} token，约 ${usd:.2f}")
    print(f"  {a.tag}（{a.model}）累计约 ${total:.2f}" + (f"（上限 ${a.max:.2f}）" if a.max is not None else ""))
    if a.max is not None and total >= a.max:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
