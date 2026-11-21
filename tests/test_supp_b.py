"""补充实验 B（claude/补充实验B_2026-11-21.md）：执行器记录与评估器同一回合、重评跳过 B5 前缀、报告能生成。"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from hyblock import supp_b

D = "m0=root q30/60; m1=m0.P1@90 q60/-30; h2=m0.P2@90; foot=h2.xp"


class TestSuppB(unittest.TestCase):
    def test_actuator_episode_matches_evaluator(self):
        from hyblock.mjcf import compile_design
        from mpc.mppi import MPPIConfig, evaluate_design
        for ter in ("flat", "slope_5"):
            t = dict(design=D, terrain=ter, physics="{}", max_modules=8)
            r = supp_b.actuator_episode(t, threads=2, steps=4)
            cfg = MPPIConfig(threads=2, episode_control_steps=4)
            e = evaluate_design(compile_design(D, max_modules=8), ter, 1, supp_b.SEED, cfg)
            self.assertAlmostEqual(r["score"], e["episodes"][0]["score"], places=2)
            for k in ("tau_cmd_sat", "tau_use", "tau_force_sat", "spd_use", "spd_fast", "at_limit"):
                self.assertTrue(0.0 <= r[k] <= 1.0, (k, r[k]))
            self.assertAlmostEqual(r["wmax_deg"], 1185.0, places=0)
        t = dict(design=D, terrain="flat", physics=json.dumps({"max_speed_deg": 370}), max_modules=8)
        self.assertAlmostEqual(supp_b.actuator_episode(t, threads=2, steps=2)["wmax_deg"], 370.0, places=0)

    def test_reeval_skips_prefix(self):
        from hyblock.reeval_best import targets
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "slope_5"
            for name in ("ga_s500", "ga_llmtyped_pre_s500"):
                d = root / name
                d.mkdir(parents=True)
                with open(d / "evals.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["gen", "idx", "status", "fitness", "design", "ep_scores"])
                    w.writeheader()
                    w.writerow(dict(gen=0, idx=0, status="mpc", fitness=10, design=D, ep_scores="[10, 10]"))
                (d / "run_meta.json").write_text(json.dumps({"args": {"terrain": "slope_5"}}), encoding="utf-8")
                (d / "DONE").write_text("x")
            T = targets([root])
            self.assertEqual([t["run"] for t in T], ["ga_s500"])
            self.assertEqual(T[0]["terrain"], "slope_5")

    def test_reports(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "actuator.csv"
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=supp_b.A_FIELDS)
                w.writeheader()
                for i in range(12):
                    w.writerow(dict(root="r", run=f"ga_s{i}", group="ga", seed=i, terrain="flat", physics="{}", design=D, nu=4,
                                    score=100 + i, env_steps=1000, terminated=False, tau_cmd_sat=0.1, tau_use=0.3,
                                    tau_force_sat=0.05, spd_use=0.1, spd_fast=0.02 * i, at_limit=0.2, wmax_deg=1185, seconds=1))
            self.assertIn("默认", supp_b.report_actuator(p))
            q = Path(td) / "cross.csv"
            with open(q, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=supp_b.C_FIELDS)
                w.writeheader()
                for i, h in enumerate(supp_b.CROSS_TERRAINS):
                    for s in range(6):
                        for e in supp_b.CROSS_TERRAINS:
                            w.writerow(dict(root=h, run=f"ga_s{s}", group="ga", seed=s, home=h, eval=e, design=f"{h}{s}",
                                            fitness=(200 if h == e else 100) + s, eps="[]", seconds=1, reused=""))
            txt = supp_b.report_cross(q)
            self.assertIn("**设计随地形专门化**", txt)


if __name__ == "__main__":
    unittest.main()
