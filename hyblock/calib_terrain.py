"""地形扩展的难度标定（claude/地形扩展计划.md 第 2 节 = 正式实验计划 变更记录第 22 条；11-13 写定，数据出来之前）。

只用随机设计（不跑任何 LLM 组）。沿用第四阶段标定 `mac_runs/calib_sc`（Mac，提交 ad28090，物理与现在相同）的
120 个文法随机设计（模块上限 8）与 3 个手工设计：
  - 平地上走不动的设计（calib_sc/flat.csv 里 3 回合平均位移 < 1 m）不再评估，记为"不通过"（地形只会更难）；
  - 平地能走的设计在每个候选难度上跑 MPC 2 回合（评估器设置与搜索相同），至少一回合达成即算通过。
通过 = 回合内躯干在 −x 方向的最远位置 ≥ mpc/terrain_ext.pass_need(地形) + 该回合的 x_start：
  沟坎 / 横梁 = 第一个障碍远侧边缘再过 0.5 m（与 hyblock/calibrate.py 相同）；
  冰面 / 斜坡 / 石块 / 不平地面 = 进入地形区 2 m（地形区从前方 1 m 开始，即位移 3 m）。
通过比例的分母 = 全部 120 个随机设计（手工设计只列出，不计入）。

事先定的选择规则（choose）：每种地形，通过比例在 [0.20, 0.30] 内的难度里取最难的一个。
没有一个在区间内 → 自动插一级再测（--auto_insert；最多一次，插入值事先定在 EXTRA 与中点规则里）：
  全部偏易（> 0.30）→ 插 EXTRA 里"更难"的一级；全部偏难（< 0.20）→ 插"更容易"的一级；
  跨过区间（相邻两级一个 > 0.30、一个 < 0.20）→ 插这两级参数的中点。
  插入后仍没有落在区间内 → 该地形不进入正式实验（dropped）。
chosen.json 同时记下每种地形的"退一级"（fallback）：已测难度里比选定的更容易、最接近的一级（没有就用 EXTRA 的"更容易"），
供 GA 试点不通过时按计划"换容易一级重新试点（最多一次）"。
平地参照（只描述）：calib_sc/flat.csv 前 2 回合中任一回合的位移（得分 × 0.05）≥ 3 m 的比例。

用法（Mac，在 robomorph 下）：
    uv run --locked python -m hyblock.calib_terrain --flat ../mac_runs/calib_sc/flat.csv --out ../mac_runs/terrain_ext/calib \
        --terrains ridged_s0.6,ridged_s0.75,ridged_s0.9,ice_0.5,ice_0.3,ice_0.2,slope_5,slope_10,slope_15,... --threads 12
    （中断后重跑会跳过已完成的行；--summary_only 只重写 summary.txt 与 chosen.json）
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from hyblock.mjcf import compile_design
from mpc.mppi import MPPIConfig, evaluate_design
from mpc.terrain_ext import parse, pass_need

CAP = 8
WALK_M = 1.0
RANGE = (0.20, 0.30)
LEVELS = {   # 计划第 1 节：每种地形的候选难度（由易到难）
    "ridged": ["ridged_s0.6", "ridged_s0.75", "ridged_s0.9"],
    "ice": ["ice_0.5", "ice_0.3", "ice_0.2"],
    "slope": ["slope_5", "slope_10", "slope_15"],
    "rocks": ["rocks_0.06", "rocks_0.1", "rocks_0.14"],
    "rough": ["rough_0.03", "rough_0.06", "rough_0.1"],
    "beams": ["beams_s1.25"],
}
EXTRA = {   # 插一级时用的值：(更容易, 更难)
    "ridged": (0.45, 1.0), "ice": (0.8, 0.1), "slope": (2.5, 20.0), "rocks": (0.03, 0.2), "rough": (0.015, 0.15),
    "beams": (1.5, 1.1),
}
PREFIX = {"ridged": "ridged_s", "beams": "beams_s", "ice": "ice_", "slope": "slope_", "rocks": "rocks_", "rough": "rough_"}
FIELDS = ["name", "modules", "terrain", "fitness", "ep_scores", "ep_progress_m", "need_m", "passed", "terminated",
          "unstable", "seconds"]


def hardness(name: str) -> float:
    """越大越难（横梁：比例越小越难，见第四阶段标定）。"""
    kind, v = parse(name)
    return {"ridged": v, "ice": -v, "slope": v, "rocks": v, "rough": v, "beams": -v}[kind]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def dt_env(xml: str) -> float:
    m = mujoco.MjModel.from_xml_string(xml)
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "hyblock_n_frames")
    nf = m.numeric_data[m.numeric_adr[i]] if i >= 0 else 5
    return float(m.opt.timestep * nf)


def evaluate(row: dict, terrain: str, episodes: int, threads: int, seed: int, steps: int) -> dict:
    xml = compile_design(row["design"], max_modules=CAP)
    t0 = time.time()
    r = evaluate_design(xml, terrain, episodes, seed, MPPIConfig(threads=threads, episode_control_steps=steps))
    dt = dt_env(xml)
    prog = [float(np.max(np.cumsum(rw)) * dt) if rw else 0.0 for rw in r["rewards"]]
    need = [pass_need(terrain) + e["x_start"] for e in r["episodes"]]
    return {"name": row["name"], "modules": row["modules"], "terrain": terrain, "fitness": round(r["fitness"], 3),
            "ep_scores": json.dumps([round(e["score"], 2) for e in r["episodes"]]),
            "ep_progress_m": json.dumps([round(p, 3) for p in prog]),
            "need_m": json.dumps([round(n, 3) for n in need]),
            "passed": int(any(p >= n for p, n in zip(prog, need))),
            "terminated": sum(e["terminated"] for e in r["episodes"]),
            "unstable": sum(e["unstable"] for e in r["episodes"]), "seconds": round(time.time() - t0, 2)}


def rates(flat: list[dict], rows: list[dict]) -> dict:
    """{terrain: (通过数, 分母 = 随机设计数, 已测数)}，只算随机设计。"""
    n_all = sum(r["name"].startswith("rand_") for r in flat)
    out = {}
    for t in sorted({r["terrain"] for r in rows}):
        rr = [r for r in rows if r["terrain"] == t and r["name"].startswith("rand_")]
        out[t] = (sum(int(r["passed"]) for r in rr), n_all, len(rr))
    return out


def tname(kind: str, v: float) -> str:
    return f"{PREFIX[kind]}{round(v, 6):g}"


def insert_level(kind: str, p: dict) -> str | None:
    """按事先定的规则给出要插的一级（p = {地形名: 比例}，都不在区间内）。"""
    order = sorted(p, key=hardness)                         # 由易到难
    if all(v > RANGE[1] for v in p.values()):
        return tname(kind, EXTRA[kind][1])
    if all(v < RANGE[0] for v in p.values()):
        return tname(kind, EXTRA[kind][0])
    for a, b in zip(order, order[1:]):                      # 跨过区间：相邻两级一个偏易、一个偏难
        if p[a] > RANGE[1] and p[b] < RANGE[0]:
            return tname(kind, (parse(a)[1] + parse(b)[1]) / 2)
    return None


def choose(rate: dict, terrains, n_walkers: int, final: bool = False) -> dict:
    """事先定的规则。只对"已测完"（已测数 = 能走的随机设计数）的难度下判断。final：插级之后，没选上的记 dropped。"""
    out = {}
    for kind in LEVELS:
        cand = [t for t in terrains if parse(t)[0] == kind]
        if not cand:
            continue
        done = [t for t in cand if t in rate and rate[t][2] >= n_walkers]
        if len(done) < len(cand):
            out[kind] = {"status": "incomplete"}
            continue
        p = {t: rate[t][0] / rate[t][1] for t in cand}
        ok = [t for t in cand if RANGE[0] <= p[t] <= RANGE[1]]
        rates_ = {t: round(v, 3) for t, v in sorted(p.items(), key=lambda kv: hardness(kv[0]))}
        if ok:
            pick = max(ok, key=hardness)
            easier = [t for t in cand if hardness(t) < hardness(pick)]
            out[kind] = {"status": "chosen", "terrain": pick, "rate": round(p[pick], 3), "rates": rates_,
                         "fallback": max(easier, key=hardness) if easier else tname(kind, EXTRA[kind][0])}
        elif final:
            out[kind] = {"status": "dropped", "rates": rates_}
        else:
            side = "too_easy" if min(p.values()) > RANGE[1] else "too_hard" if max(p.values()) < RANGE[0] else "straddles"
            out[kind] = {"status": side, "rates": rates_, "insert": insert_level(kind, p)}
    return out


def flat_reference(flat: list[dict]) -> float:
    rnd = [r for r in flat if r["name"].startswith("rand_")]
    ok = [any(s * 0.05 >= 3.0 for s in json.loads(r["ep_scores"])[:2]) for r in rnd]
    return sum(ok) / len(rnd)


def write_summary(out: Path, flat: list[dict], terrains, episodes: int = 2, final: bool = False) -> str:
    rows = _read(out / "terrain.csv")
    walkers = [r for r in flat if float(r["distance_m"]) >= WALK_M]
    n_w = sum(r["name"].startswith("rand_") for r in walkers)
    rate = rates(flat, rows)
    ch = choose(rate, terrains, n_w, final)
    L = [f"# 地形扩展：难度标定（{time.strftime('%Y-%m-%d %H:%M')}）", "",
         f"随机设计 {sum(r['name'].startswith('rand_') for r in flat)} 个（calib_sc），平地能走的 {n_w} 个参加评估；"
         f"每个 {episodes} 回合，至少一回合通过即算通过。目标区间 {RANGE[0]:.2f}–{RANGE[1]:.2f}，区间内取最难的。", "",
         f"平地参照（2 回合任一回合位移 ≥ 3 m）：{flat_reference(flat):.2f}", "",
         "| 地形 | 通过 / 随机设计数 | 比例 | 已测 | 能走设计的得分中位数 | 每设计用时中位数 s |", "|---|---|---|---|---|---|"]
    for t in sorted(rate, key=lambda t: (parse(t)[0], hardness(t))):
        rr = [r for r in rows if r["terrain"] == t and r["name"].startswith("rand_")]
        L.append(f"| {t} | {rate[t][0]} / {rate[t][1]} | {rate[t][0] / rate[t][1]:.2f} | {rate[t][2]} / {n_w} | "
                 f"{np.median([float(r['fitness']) for r in rr]):.1f} | {np.median([float(r['seconds']) for r in rr]):.1f} |")
    L += ["", "## 按事先规则的选择", ""]
    for kind, c in ch.items():
        if c["status"] == "chosen":
            L.append(f"- {kind}：**{c['terrain']}**（{c['rate']:.2f}；退一级 = {c['fallback']}）")
        elif c["status"] == "incomplete":
            L.append(f"- {kind}：还没测完")
        elif c["status"] == "dropped":
            L.append(f"- {kind}：插级后仍没有难度落在区间内（{c['rates']}）→ **不进入正式实验**")
        else:
            L.append(f"- {kind}：没有难度落在区间内（{c['status']}：{c['rates']}）→ 插一级 {c['insert']}")
    hand = [r for r in rows if not r["name"].startswith("rand_")]
    if hand:
        L += ["", "手工设计（不计入比例）：" + "；".join(f"{r['name']} {r['terrain']} {'过' if r['passed'] == '1' else '不过'}"
                                               f"（{r['ep_progress_m']} / {r['need_m']} m）" for r in hand)]
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    (out / "chosen.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "chosen.txt").write_text("".join(f"{c['terrain']} {c['fallback']}\n" for c in ch.values()
                                            if c["status"] == "chosen"), encoding="utf-8")   # 给 shell 用：难度 退一级
    return text


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--flat", required=True, help="calib_sc/flat.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--terrains", default=",".join(t for v in LEVELS.values() for t in v))
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--mpc_seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--limit", type=int, default=None, help="离线测试用：只评前几个设计")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--auto_insert", action="store_true", help="没有难度落在区间内的地形自动插一级再测（最多一次）")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    terrains = [t for t in a.terrains.split(",") if t]
    for t in terrains:
        if parse(t) is None:
            raise SystemExit(f"不是扩展地形名：{t}")
    flat = _read(Path(a.flat))
    walkers = [r for r in flat if float(r["distance_m"]) >= WALK_M][:a.limit]

    def run(ts):
        done = {(r["name"], r["terrain"]) for r in _read(out / "terrain.csv")}
        for t in ts:                            # 按地形依次跑：中途停下也能先拿到完整的几级
            for r in walkers:
                if (r["name"], t) in done:
                    continue
                row = evaluate(r, t, a.episodes, a.threads, a.mpc_seed, a.steps)
                _append(out / "terrain.csv", row)
                print(f"[calib] {t} {r['name']} progress {row['ep_progress_m']} need {row['need_m']} "
                      f"passed={row['passed']} {row['seconds']:.0f}s", flush=True)
            print(write_summary(out, flat, terrains, a.episodes), flush=True)

    if not a.summary_only:
        run(terrains)
    final = False
    if a.auto_insert:
        prev = out / "inserted.txt"             # 第一次算出的插入级记下来，续跑时沿用（最多插一次）
        if prev.exists():
            extra = [t for t in prev.read_text(encoding="utf-8").split() if t]
        else:
            write_summary(out, flat, terrains, a.episodes)
            ch = json.loads((out / "chosen.json").read_text(encoding="utf-8"))
            extra = [c["insert"] for c in ch.values() if c.get("insert") and c["insert"] not in terrains]
            prev.write_text("".join(t + "\n" for t in extra), encoding="utf-8")
        terrains = terrains + [t for t in extra if t not in terrains]
        if extra and not a.summary_only:
            run(extra)
        final = True
    print(write_summary(out, flat, terrains, a.episodes, final))


if __name__ == "__main__":
    main()
