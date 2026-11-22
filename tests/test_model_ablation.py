"""换模型消融（正式实验计划变更记录第 23 条）：分析脚本、模型列表、--model 同时传给 LLM 变异与 ②。"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from hyblock import model_ablation_analysis as MA

FIELDS = ["gen", "idx", "status", "fitness", "mpc_s", "llm_in", "llm_out", "sel_slot", "sel_pred"]


def fake_run(d: Path, seed: int, shift: float, model: str = "claude-haiku-5-5"):
    d.mkdir(parents=True)
    rows = []
    for g in range(13):
        for i in range(16):
            f = 10.0 + i if g == 0 else 10.0 + i + g * 5 + shift
            rows.append(dict(gen=g, idx=i, status="mpc", fitness=f, mpc_s=1, llm_in=100, llm_out=50, sel_slot="", sel_pred=""))
    with open(d / "evals.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (d / "run_meta.json").write_text(json.dumps({"args": {"model": model, "seed": seed}}), encoding="utf-8")
    (d / "DONE").write_text("x")


class TestModelAblation(unittest.TestCase):
    def test_analysis(self):
        with tempfile.TemporaryDirectory() as t:
            base, root = Path(t) / "base", Path(t) / "root"
            for s in (500, 501, 502):
                fake_run(base / f"ga_s{s}", s, 0)
                fake_run(base / f"ga_llmtyped_sel_llmEv_s{s}", s, 20)
                fake_run(root / f"ga_llmtyped_sel_llmEv_sonnet_s{s}", s, 30 + s - 500, "claude-sonnet-5-5")
            txt = MA.analyse(base, root, ["haiku", "sonnet"], [500, 501, 502, 503])
            self.assertIn("模型：`claude-sonnet-5-5`", txt)
            self.assertIn("| B6′ − B6 后期中位数 | 3 | +11.0 |", txt)      # (30+31+32) − 3×20 = 33 / 3
            self.assertIn("| B6′ − B1 后期中位数 | 3 | +31.0 |", txt)
            self.assertIn("| B6′ − B6 后期中位数 | 3 | +0.0 |", txt)       # haiku 自检
            self.assertIn("种子 503（未开始）", txt)
            self.assertIn("第 0 代中位数与 B1 相同的种子 3 / 3", txt)
            self.assertIn("**高于 Haiku**", txt)

    def test_models_file(self):
        p = Path(__file__).resolve().parents[1] / "wj_tools" / "model_ablation_models.txt"
        lines = [l.split() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        self.assertTrue(lines)
        for parts in lines:
            self.assertEqual(len(parts), 2, parts)
        self.assertEqual(lines[0], ["sonnet", "claude-sonnet-5-5"])

    def test_model_reaches_mutation_and_selection(self):
        from hyblock import search
        from hyblock.selection import make_selector
        from hyblock.llm_mutation import LLMMutator
        with tempfile.TemporaryDirectory() as t:
            a = search.build_parser().parse_args(["--method", "ga", "--mutation", "llm_typed", "--select", "llm",
                                                  "--model", "claude-sonnet-5-5", "--terrain", "slope_5",
                                                  "--out", t, "--seed", "500"])
            llm = search.LLM(a, Path(t))
            sel = make_selector(a, Path(t), llm)
            mut = LLMMutator(a, Path(t), llm)
            self.assertIs(sel.llm, llm)
            self.assertEqual(llm.a.model, "claude-sonnet-5-5")
            k_s = llm._key("u", 5, 0, "sys", "mut0")
            a.model = "claude-haiku-5-5"
            self.assertNotEqual(k_s, llm._key("u", 5, 0, "sys", "mut0"))   # 缓存按模型分开
            self.assertIsNotNone(mut)


if __name__ == "__main__":
    unittest.main()


class TestAblationCost(unittest.TestCase):
    """分批的费用上限（11-22）：只数真正调用 API 的行；B6′ 的第 0–4 代是复制来的，不算。"""

    def _csv(self, path: Path, fields, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def test_cost(self):
        from hyblock import ablation_cost as AC
        mut = ["gen", "idx", "llm_in", "llm_out", "cached"]
        sel = ["gen", "vote", "llm_in", "llm_out", "cached"]
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            # 前缀：2 次调用（各 1000 / 100）+ 1 次缓存（不算）
            self._csv(root / "ga_llmtyped_pre_sonnet_s500" / "llm_mut.csv", mut, [
                dict(gen=1, idx=0, llm_in=1000, llm_out=100, cached="False"),
                dict(gen=4, idx=0, llm_in=1000, llm_out=100, cached="False"),
                dict(gen=4, idx=1, llm_in=1000, llm_out=100, cached="True")])
            # B6′：第 4 代是复制来的（不算），第 5 代 1 次调用；② 1 次调用；空 token 的行不算
            self._csv(root / "ga_llmtyped_sel_llmEv_sonnet_s500" / "llm_mut.csv", mut, [
                dict(gen=4, idx=0, llm_in=1000, llm_out=100, cached="False"),
                dict(gen=5, idx=0, llm_in=2000, llm_out=200, cached="False"),
                dict(gen=6, idx=0, llm_in="", llm_out="", cached="")])
            self._csv(root / "ga_llmtyped_sel_llmEv_sonnet_s500" / "select.csv", sel, [
                dict(gen=5, vote=0, llm_in=5000, llm_out=500, cached="False")])
            self.assertEqual(AC.seed_tokens(root, "sonnet", 500), (9000, 900, 4))
            total, rows = AC.cost(root, "sonnet", AC.PRICES["claude-sonnet-5-5"])
            self.assertAlmostEqual(total, 9000 * 2e-6 + 900 * 10e-6)      # 0.027
            self.assertEqual([r[0] for r in rows], [500])
            args = ["--root", t, "--tag", "sonnet", "--model", "claude-sonnet-5-5"]
            self.assertEqual(AC.main(args + ["--max", "1"]), 0)
            self.assertEqual(AC.main(args + ["--max", "0.02"]), 3)
            self.assertEqual(AC.main(["--root", t, "--tag", "x", "--model", "unknown-model"]), 2)

    def test_batch_file(self):
        p = Path(__file__).resolve().parents[1] / "wj_tools" / "model_ablation_batch.txt"
        kv = dict(l.split("=", 1) for l in p.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#"))
        from hyblock import ablation_cost as AC
        m = Path(__file__).resolve().parents[1] / "wj_tools" / "model_ablation_models.txt"
        models = dict(l.split() for l in m.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#"))
        for k, v in kv.items():
            key, _, tag = k.partition("_") if k.startswith("SEEDS") else (k[:7], "_", k[8:])
            self.assertIn(key, ("SEEDS", "MAX_USD"), k)
            if tag:
                self.assertIn(tag, models, k)                     # 标签必须在模型列表里
                self.assertIn(models[tag], AC.PRICES, k)          # 有上限的模型必须知道单价
            if key == "SEEDS":
                seeds = [int(x) for x in v.split()]
                self.assertTrue(seeds)
                self.assertTrue(all(500 <= s <= 509 for s in seeds))
                self.assertEqual(len(seeds), len(set(seeds)))
            else:
                self.assertGreater(float(v), 0)
