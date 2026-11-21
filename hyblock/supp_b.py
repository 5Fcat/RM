"""补充实验 B（`claude/补充实验B_2026-11-21.md`；只描述，结果出来之前写定）。

  actuator：执行器使用情况（B1）——各运行的最终最好设计在同一评估器下跑 1 回合（回合种子 1000，= 重评的第 1 回合），
            每个环境步记录各关节的力矩指令、实际力矩、角速度、关节角，汇总成力矩饱和、速度利用、关节限位的比例
  cross   ：跨地形交叉评估（B3）——平地 / 横梁 s1.0 / 沟坎 s0.75 / 斜坡 5° / 横梁 s1.1 各自找到的最终最好设计
            （GA 与 LLM 变异 + ②）放到全部 5 种地形上评估（2 回合，回合种子 1000、1001）
  （B2 = 最优设计重评的补充，用 hyblock.reeval_best run，不在这里）

用法：python -m hyblock.supp_b actuator --roots <根目录>... --out <目录> [--threads 12]
      python -m hyblock.supp_b cross --roots <根目录>... --out <目录> [--threads 12]
      python -m hyblock.supp_b report-actuator|report-cross --out <目录>
都可中断，再运行从断点继续（csv 逐行追加）。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

SEED = 1
CROSS_TERRAINS = ["flat", "beams", "ridged_s0.75", "slope_5", "beams_s1.1"]
CROSS_GROUPS = ("ga", "ga_llmtyped_sel_llmEv")
SAT = 0.95          # |动作| ≥ 0.95 = 力矩指令饱和
FAST = 0.5          # |角速度| ≥ 0.5 × 速度上限
LIM_DEG = 2.0       # 关节角距离范围端点 ≤ 2° = 顶到限位
A_FIELDS = ["root", "run", "group", "seed", "terrain", "physics", "design", "nu", "score", "env_steps", "terminated",
            "tau_cmd_sat", "tau_use", "tau_force_sat", "spd_use", "spd_fast", "at_limit", "wmax_deg", "seconds"]
C_FIELDS = ["root", "run", "group", "seed", "home", "eval", "design", "fitness", "eps", "seconds", "reused"]


def _model(t: dict, terrain: str, threads: int):
    from hyblock.mjcf import compile_design
    from hyblock.reeval_best import physics_kw
    from mpc import terrain_ext
    from mpc.mppi import MPPIConfig, MPPIEvaluator, load_model
    cfg = MPPIConfig(threads=threads, episode_control_steps=t.get("steps", 250), terrain_scale=t.get("terrain_scale", 1.0))
    xml = compile_design(t["design"], max_modules=t.get("max_modules", 8), **physics_kw(json.loads(t["physics"])))
    model = load_model(xml, terrain, cfg.mjx_solver, cfg.terrain_scale, cfg.min_solver_iterations)
    ev = MPPIEvaluator(model, cfg, seed=SEED)
    ev.ground = terrain_ext.ground_height(terrain)
    return cfg, model, ev


def actuator_episode(t: dict, threads: int = 12, steps: int | None = None) -> dict:
    """与 MPPIEvaluator.run_episode 相同的回合循环，多记录关节量。"""
    import mujoco
    if steps is not None:
        t = dict(t, steps=steps)
    cfg, m, ev = _model(t, t["terrain"], threads)
    jid = m.actuator_trnid[:, 0]
    dof, qadr = m.jnt_dofadr[jid], m.jnt_qposadr[jid]
    lo_r, hi_r = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    flo, fhi = m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1]
    wmax = ev.ctrl_hi / m.dof_damping[dof]                     # mjcf：阻尼 = 伸出侧力矩上限 / 最大速度
    lim = np.radians(LIM_DEG)
    t0 = time.time()
    d = ev.reset(SEED * 1000)
    plan, history = np.zeros((cfg.horizon, ev.nu)), np.zeros((0, ev.nu))
    rewards, A, F, V, L = [], [], [], [], []
    terminated = False
    badqacc = int(mujoco.mjtWarning.mjWARN_BADQACC)
    try:
        for k in range(cfg.episode_control_steps):
            plan = ev.update(d, plan, history, cfg.warmup_samples if k == 0 else cfg.num_samples)
            a = plan[0].copy()
            d.ctrl[:] = ev.scale(a)
            for _ in range(cfg.interval):
                x0, nwarn = d.qpos[0], d.warning[badqacc].number
                mujoco.mj_step(m, d, nstep=ev.n_frames)
                r = -(d.qpos[0] - x0) / ev.dt_env
                if d.warning[badqacc].number > nwarn or not np.isfinite(r) or abs(r) > cfg.max_speed:
                    terminated = True
                    break
                rewards.append(r)
                f = d.actuator_force.copy()
                A.append(np.abs(a))
                F.append(np.where(f >= 0, f / fhi, f / flo))      # 各侧上限的比例（0–1）
                V.append(np.abs(d.qvel[dof]) / wmax)
                q = d.qpos[qadr]
                L.append((q - lo_r <= lim) | (hi_r - q <= lim))
                if ev.unhealthy(d.qpos[:7]):
                    terminated = True
                    break
            history = np.vstack([history, a[None]])[-cfg.horizon:]
            plan = np.vstack([plan[1:], np.zeros((1, ev.nu))])
            if terminated:
                break
    finally:
        ev.close()
    A, F, V, L = (np.array(x) if x else np.zeros((0, ev.nu)) for x in (A, F, V, L))
    mean = lambda x: float(np.mean(x)) if x.size else float("nan")
    return dict(nu=int(ev.nu), score=round(float(np.sum(rewards)), 3), env_steps=len(rewards), terminated=terminated,
                tau_cmd_sat=round(mean(A >= SAT), 4), tau_use=round(mean(np.clip(F, 0, None)), 4),
                tau_force_sat=round(mean(F >= SAT), 4), spd_use=round(mean(V), 4), spd_fast=round(mean(V >= FAST), 4),
                at_limit=round(mean(L), 4), wmax_deg=round(float(np.degrees(np.median(wmax))), 1),
                seconds=round(time.time() - t0, 1))


def _resume(path: Path, key) -> dict:
    have = {}
    if path.exists():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            have[key(r)] = r
    return have


def cmd_actuator(a):
    from hyblock.reeval_best import targets
    T = targets([Path(r) for r in a.roots])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "actuator.csv"
    have = _resume(path, lambda r: (r["root"], r["run"]))
    print(f"执行器使用情况：{len(T)} 个最终最好设计，各 1 回合（回合种子 {SEED * 1000}）", flush=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=A_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, t in enumerate(T):
            if (t["root"], t["run"]) in have:
                continue
            t.update(actuator_episode(t, a.threads))
            w.writerow(t)
            f.flush()
            print(f"[{i + 1}/{len(T)}] {t['root']}/{t['run']}: 得分 {t['score']:.0f}，力矩饱和 {t['tau_cmd_sat']:.2f}，"
                  f"速度 ≥ {FAST} 上限 {t['spd_fast']:.2f}（{t['seconds']} s）", flush=True)
    (out / "report_actuator.md").write_text(report_actuator(path), encoding="utf-8")


def cmd_cross(a):
    from hyblock.mjcf import compile_design
    from hyblock.reeval_best import physics_kw, targets
    from mpc.mppi import MPPIConfig, evaluate_design
    T = [t for t in targets([Path(r) for r in a.roots]) if t["group"] in CROSS_GROUPS and t["physics"] == "{}"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cross.csv"
    have = _resume(path, lambda r: (r["root"], r["run"], r["eval"]))
    cache = {(r["design"], r["eval"]): r for r in have.values()}
    print(f"跨地形交叉评估：{len(T)} 个最终最好设计 × {len(CROSS_TERRAINS)} 种地形，各 2 回合", flush=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=C_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, t in enumerate(T):
            for ter in CROSS_TERRAINS:
                if (t["root"], t["run"], ter) in have:
                    continue
                row = dict(root=t["root"], run=t["run"], group=t["group"], seed=t["seed"], home=t["terrain"], eval=ter,
                           design=t["design"])
                c = cache.get((t["design"], ter))
                if c:
                    row.update(fitness=c["fitness"], eps=c["eps"], seconds=0, reused=c["run"])
                else:
                    t0 = time.time()
                    cfg = MPPIConfig(threads=a.threads)
                    r = evaluate_design(compile_design(t["design"], max_modules=t["max_modules"]), ter, 2, SEED, cfg)
                    row.update(fitness=round(r["fitness"], 3), eps=json.dumps([round(e["score"], 2) for e in r["episodes"]]),
                               seconds=round(time.time() - t0, 1), reused="")
                w.writerow(row)
                f.flush()
                cache[(t["design"], ter)] = row
                have[(t["root"], t["run"], ter)] = row
            print(f"[{i + 1}/{len(T)}] {t['root']}/{t['run']}（{t['terrain']}）完成", flush=True)
    (out / "report_cross.md").write_text(report_cross(path), encoding="utf-8")


def _cond(r) -> str:
    ph = json.loads(r["physics"]) if r.get("physics") else {}
    if "max_speed_deg" in ph:
        return f"速度 {ph['max_speed_deg']:g}"
    if "pressure" in ph:
        return f"压力 {ph['pressure']:g} MPa"
    if "mass_scale" in ph:
        return f"质量 ×{ph['mass_scale']:g}"
    return "默认"


def report_actuator(path: Path) -> str:
    R = list(csv.DictReader(open(path, encoding="utf-8")))
    keys = [("tau_cmd_sat", "力矩指令饱和（|动作| ≥ 0.95）"), ("tau_force_sat", "实际力矩 ≥ 0.95 上限"),
            ("tau_use", "实际力矩 / 上限 均值"), ("spd_fast", f"角速度 ≥ {FAST} × 速度上限"), ("spd_use", "角速度 / 速度上限 均值"),
            ("at_limit", f"关节角在限位 {LIM_DEG:g}° 以内")]
    L = ["# 补充实验 B1：执行器使用情况（只描述）", "",
         "登记：`claude/补充实验B_2026-11-21.md`。每个运行的最终最好设计跑 1 回合（回合种子 1000），按环境步 × 关节平均。", "",
         "## 按条件（地形 × 物理参数）", "",
         "| 条件 | 设计数 | 得分均值 | " + " | ".join(k[1] for k in keys) + " |", "|---|---|---|" + "---|" * len(keys)]
    conds = {}
    for r in R:
        conds.setdefault((r["terrain"], _cond(r)), []).append(r)
    for (ter, c), S in conds.items():
        L.append(f"| {ter}，{c} | {len(S)} | {np.mean([float(r['score']) for r in S]):.0f} | "
                 + " | ".join(f"{np.mean([float(r[k]) for r in S]):.3f}" for k, _ in keys) + " |")
    L += ["", "## 按组（只看默认物理参数）", "", "| 组 | 设计数 | " + " | ".join(k[1] for k in keys) + " |", "|---|---|" + "---|" * len(keys)]
    grp = {}
    for r in R:
        if _cond(r) == "默认":
            grp.setdefault(r["group"], []).append(r)
    for g, S in grp.items():
        L.append(f"| {g} | {len(S)} | " + " | ".join(f"{np.mean([float(r[k]) for r in S]):.3f}" for k, _ in keys) + " |")
    from scipy.stats import spearmanr
    D = [r for r in R if _cond(r) == "默认"]
    if len(D) >= 10:
        L += ["", "## 与得分的关系（默认物理参数，全部设计）", ""]
        for k, name in keys:
            L.append(f"- Spearman（{name}，该回合得分）= {spearmanr([float(r[k]) for r in D], [float(r['score']) for r in D])[0]:+.2f}")
    L += ["", f"- 速度上限 = 各关节 伸出侧力矩上限 / 关节阻尼（`hyblock/mjcf.py` 的线性力矩–速度模型；默认约 1185 deg/s）；"
          "空载时角速度才能接近上限，所以\"≥ 0.5 × 上限\"已经是很高的利用率"]
    return "\n".join(L) + "\n"


def report_cross(path: Path) -> str:
    R = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in R:
        r["f"] = float(r["fitness"])
    homes = [h for h in CROSS_TERRAINS if any(r["home"] == h for r in R)]
    L = ["# 补充实验 B3：跨地形交叉评估（只描述）", "",
         "登记：`claude/补充实验B_2026-11-21.md`。行 = 设计在哪种地形上找到（各种子的最终最好，GA 与 LLM 变异 + ②），"
         "列 = 在哪种地形上评估（2 回合，回合种子 1000、1001）。", "",
         "## 得分均值", "", "| 设计来自 ＼ 评估于 | " + " | ".join(CROSS_TERRAINS) + " |", "|---|" + "---|" * len(CROSS_TERRAINS)]
    M = {}
    for h in homes:
        cells = []
        for e in CROSS_TERRAINS:
            v = [r["f"] for r in R if r["home"] == h and r["eval"] == e]
            M[(h, e)] = float(np.mean(v)) if v else float("nan")
            cells.append(f"{M[(h, e)]:.0f}" + ("（本地形）" if h == e else ""))
        L.append(f"| {h}（{sum(r['home'] == h and r['eval'] == h for r in R)} 个） | " + " | ".join(cells) + " |")
    L += ["", "## 本地形优势：在每种地形上，本地形找到的设计 vs 其他地形找到的设计", "",
          "| 评估于 | 本地形设计均值 | 其他地形设计均值 | 比 | 本地形设计在全部设计里的平均百分位 |", "|---|---|---|---|---|"]
    nwin = 0
    for e in homes:
        own = [r["f"] for r in R if r["eval"] == e and r["home"] == e]
        oth = [r["f"] for r in R if r["eval"] == e and r["home"] != e]
        allv = np.array([r["f"] for r in R if r["eval"] == e])
        pct = np.mean([np.mean(allv < v) for v in own]) if own else float("nan")
        nwin += np.mean(own) > np.mean(oth)
        L.append(f"| {e} | {np.mean(own):.0f} | {np.mean(oth):.0f} | {np.mean(own) / np.mean(oth):.2f} | {pct:.2f} |")
    L += ["", f"事先定的读法：本地形设计均值高于其他地形设计的地形 {nwin} / {len(homes)} → "
          + ("**设计随地形专门化**" if nwin >= len(homes) - 1 else "**部分专门化**" if nwin >= len(homes) / 2 else "**看不出专门化**")]
    from scipy.stats import spearmanr
    L += ["", "## 设计在两种地形上的得分是否相关（全部设计，Spearman）", "", "| | " + " | ".join(CROSS_TERRAINS) + " |",
          "|---|" + "---|" * len(CROSS_TERRAINS)]
    by = {}
    for r in R:
        by.setdefault((r["root"], r["run"]), {})[r["eval"]] = r["f"]
    for e1 in CROSS_TERRAINS:
        row = []
        for e2 in CROSS_TERRAINS:
            pairs = [(v[e1], v[e2]) for v in by.values() if e1 in v and e2 in v]
            row.append(f"{spearmanr(*zip(*pairs))[0]:+.2f}" if len(pairs) >= 5 and e1 != e2 else "—")
        L.append(f"| {e1} | " + " | ".join(row) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("actuator", "cross"):
        q = sub.add_parser(name)
        q.add_argument("--roots", nargs="+", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--threads", type=int, default=12)
    for name in ("report-actuator", "report-cross"):
        sub.add_parser(name).add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "actuator":
        cmd_actuator(a)
    elif a.cmd == "cross":
        cmd_cross(a)
    elif a.cmd == "report-actuator":
        print(report_actuator(Path(a.out) / "actuator.csv"))
    else:
        print(report_cross(Path(a.out) / "cross.csv"))


if __name__ == "__main__":
    main()
