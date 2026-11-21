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
