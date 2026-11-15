"""地形扩展（claude/地形扩展计划.md = 正式实验计划 变更记录第 22 条，11-13）：
旧地形逐字节不变；新地形的几何、摩擦、布局固定、起点平坦、地面高度函数、终止判定、LLM 地形描述、标定与分析脚本。"""
import csv
import hashlib
import json
import math
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from hyblock import llm_mutation, selection
from hyblock.calibrate import HAND
from hyblock.design import to_text
from hyblock.generator import random_design
from hyblock.mjcf import compile_design
from hyblock.terrain_text import NAME, TERRAIN, adapt
from mpc import terrain_ext as te
from mpc.mppi import MPPIConfig, MPPIEvaluator, apply_terrain, evaluate_design, load_model

QUAD = compile_design(HAND["hand_quad"], max_modules=8)
# apply_terrain(hand_quad) 的 sha256，用 11-13 改动之前的 mpc/mppi.py（提交 2a8127a）算出
OLD_SHA = {("flat", 1.0): "fc13888946ad88a1a9e16329d7d49193092639d752a865eac47a185d6e883cb2",
           ("ridged", 1.0): "97992e56908bc1db90ca501304787347df8e3df50ed42a55d12961893f2803a2",
           ("beams", 1.0): "adccb64da0ee60c5000bbdefe653e561094a9ff355fae827bc06cd986a8a51c1",
           ("frozen_lake", 1.0): "401ab722db426f26180cc47f699049f143de2be02d780c0a48faa8fd7ed29a60",
           ("beams", 1.25): "2682d999b8798a7e5bc8b9aee46db91c2ae21ea617fe34e1145dc73c9cabff5d",
           ("ridged", 0.75): "34db1bc4ab03950ef79ae6722307fb38b22a73283b31db6fcc5bed0bb26f6507"}
NEW = ["ridged_s0.6", "ridged_s0.75", "ridged_s0.9", "ice_0.5", "ice_0.3", "ice_0.2", "slope_5", "slope_10",
       "slope_15", "rocks_0.06", "rocks_0.1", "rocks_0.14", "rough_0.03", "rough_0.06", "rough_0.1", "beams_s1.25"]


def model(name, xml=QUAD):
    m = mujoco.MjModel.from_xml_string(apply_terrain(xml, name))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def ray_z(m, d, x, y):
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(m, d, np.array([x, y, 30.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
    return 30.0 - dist


class OldTerrainsUnchanged(unittest.TestCase):
    def test_xml_byte_identical(self):
        for (t, s), sha in OLD_SHA.items():
            self.assertEqual(hashlib.sha256(apply_terrain(QUAD, t, s).encode()).hexdigest(), sha, (t, s))

    def test_old_names_not_extended(self):
        for t in ("flat", "ridged", "beams", "frozen_lake"):
            self.assertIsNone(te.parse(t))
            self.assertIsNone(te.ground_height(t))

    def test_beams_s_equals_scaled_beams(self):
        self.assertEqual(apply_terrain(QUAD, "beams_s1.25"), apply_terrain(QUAD, "beams", 1.25))
        self.assertEqual(apply_terrain(QUAD, "beams_s1"), apply_terrain(QUAD, "beams", 1.0))

    def test_scale_must_be_one(self):
        with self.assertRaises(ValueError):
            apply_terrain(QUAD, "slope_10", 1.25)


class ParseTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(te.parse("ridged_s0.75"), ("ridged", 0.75))
        self.assertEqual(te.parse("beams_s1.25"), ("beams", 1.25))
        self.assertEqual(te.parse("ice_0.3"), ("ice", 0.3))
        self.assertEqual(te.parse("slope_10"), ("slope", 10.0))
        self.assertEqual(te.parse("rocks_0.1"), ("rocks", 0.1))
        self.assertEqual(te.parse("rough_0.06"), ("rough", 0.06))
        for bad in ("slope", "slope_x", "rocks_-1", "sloped_10", "flat_1"):
            self.assertIsNone(te.parse(bad), bad)
        for bad in ("slope_50", "ice_3", "beams_s0.8", "rocks_0"):
            with self.assertRaises(ValueError):
                te.parse(bad)

    def test_pass_need(self):
        self.assertAlmostEqual(te.pass_need("ridged_s1"), 1.7)         # = hyblock/calibrate.py at scale 1
        self.assertAlmostEqual(te.pass_need("beams_s1.25"), 2.0)       # = hyblock/calibrate.py at scale 1.25
        self.assertAlmostEqual(te.pass_need("ridged_s0.6"), 1.0 + 0.12 + 0.5)
        for t in ("ice_0.3", "slope_10", "rocks_0.1", "rough_0.06"):
            self.assertAlmostEqual(te.pass_need(t), 3.0)


class GeometryTests(unittest.TestCase):
    def test_all_compile_and_deterministic(self):
        for t in NEW:
            a, b = apply_terrain(QUAD, t), apply_terrain(QUAD, t)
            self.assertEqual(a, b, t)
            mujoco.MjModel.from_xml_string(a)

    def test_slope(self):
        for deg in (5, 10, 15):
            m, d = model(f"slope_{deg}")
            g = te.ground_height(f"slope_{deg}")
            self.assertLess(abs(ray_z(m, d, -0.9, 0.3)), 1e-6)                       # flat before 1 m
            z1, z2 = ray_z(m, d, -5.0, 0.0), ray_z(m, d, -15.0, 2.0)
            self.assertAlmostEqual(math.degrees(math.atan((z2 - z1) / 10.0)), deg, places=4)
            for x, y in ((-1.5, 0.0), (-20.0, -3.0), (-38.0, 5.0)):
                self.assertAlmostEqual(ray_z(m, d, x, y), float(g(x, y)), places=5)

    def test_rocks(self):
        lay6, lay14 = te.rocks_layout(0.06), te.rocks_layout(0.14)
        np.testing.assert_array_equal(lay6[:, [0, 1, 2, 3, 5]], lay14[:, [0, 1, 2, 3, 5]])   # same layout
        n = len(lay14)
        self.assertEqual(n, round(te.ROCK_DENSITY * (te.START_M + te.LENGTH_M - te.ROCK_START_M) * 2 * te.HALF_WIDTH_M))
        self.assertTrue(np.all((lay14[:, 4] >= 0.07 - 1e-9) & (lay14[:, 4] <= 0.14 + 1e-9)))
        self.assertTrue(np.all(lay14[:, 0] <= -te.ROCK_START_M))
        m, d = model("rocks_0.14")
        rocks = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("rock_")]
        self.assertEqual(len(rocks), n)
        top = [d.geom_xpos[g][2] + m.geom_size[g][2] for g in rocks]
        self.assertAlmostEqual(max(top), lay14[:, 4].max(), places=5)
        edge = [d.geom_xpos[g][0] + math.hypot(m.geom_size[g][0], m.geom_size[g][1]) for g in rocks]
        self.assertLess(max(edge), -1.0)                                          # nothing within 1 m
        x, y, hx, hy, h, yaw = lay14[0]
        self.assertAlmostEqual(ray_z(m, d, x, y), h, places=5)

    def test_rough(self):
        for a in (0.03, 0.06, 0.1):
            m, d = model(f"rough_{a}")
            g = te.ground_height(f"rough_{a}")
            xs, ys, H = te.rough_grid(a)
            self.assertAlmostEqual(H.max(), a)
            self.assertAlmostEqual(H.min(), 0.0)
            self.assertTrue(np.all(H[:, 0] == 0.0))                              # starts at 0 height (ramp)
            rng = np.random.default_rng(1)
            for x, y in zip(rng.uniform(-40.9, -1.05, 60), rng.uniform(-3.9, 3.9, 60)):
                self.assertLess(abs(ray_z(m, d, x, y) - float(g(x, y))), 0.002)
            self.assertEqual(float(g(-0.5, 0.0)), 0.0)

    def test_ridged_scaled(self):
        for s in (0.6, 0.75, 0.9):
            m, d = model(f"ridged_s{s}")
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
            ridges = [g for g, n in enumerate(names) if n.startswith("ridge_")]
            xs = sorted((d.geom_xpos[g][0] for g in ridges), reverse=True)
            self.assertAlmostEqual(xs[0], -1.0, places=5)                         # first ridge at 1 m
            self.assertAlmostEqual(xs[0] - xs[1], 2.0 * s, places=5)
            self.assertLessEqual(xs[-1], -1.0 - te.LENGTH_M + 2.0 * s)
            self.assertAlmostEqual(m.geom_size[ridges[0]][0], 0.2 * s)
            self.assertAlmostEqual(ray_z(m, d, xs[1], 0.0), 0.2 * s, places=5)

    def test_ice_friction(self):
        for mu in (0.5, 0.3, 0.2):
            m = mujoco.MjModel.from_xml_string(apply_terrain(QUAD, f"ice_{mu}"))
            d = mujoco.MjData(m)
            for _ in range(300):
                mujoco.mj_step(m, d)
            fr = {round(float(d.contact[i].friction[0]), 6) for i in range(d.ncon)}
            self.assertEqual(fr, {mu})

    def test_terrain_copies_floor_contact_settings(self):
        for t in ("slope_10", "rocks_0.1", "rough_0.06", "ridged_s0.75"):
            m, _ = model(t)
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == 0 and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
                    self.assertEqual(m.geom_priority[g], 1, t)
                    self.assertAlmostEqual(m.geom_friction[g][0], 1.0)

    def test_start_is_clear(self):
        """初始姿态（MPC reset，含关节噪声）下机器人不碰任何地形几何（地面除外）。"""
        rng = random.Random(0)
        xmls = [QUAD]
        while len(xmls) < 12:
            try:
                xmls.append(compile_design(to_text(random_design(rng, max_modules=8)), max_modules=8))
            except Exception:
                pass
        for t in NEW:
            for xml in xmls:
                m = load_model(xml, t)
                ev = MPPIEvaluator(m, MPPIConfig(threads=1))
                ev.ground = te.ground_height(t)
                d = ev.reset(0)
                mujoco.mj_forward(m, d)
                for i in range(d.ncon):
                    c = d.contact[i]
                    for g in (c.geom1, c.geom2):
                        if m.geom_bodyid[g] == 0:
                            self.assertEqual(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), "floor", t)
                ev.close()


class GroundRelativeTermination(unittest.TestCase):
    def test_slope_torso_height_relative(self):
        m = load_model(QUAD, "slope_10")
        ev = MPPIEvaluator(m, MPPIConfig(threads=1))
        ev.ground = te.ground_height("slope_10")
        d = ev.reset(0)
        pose = d.qpos[:7].copy()
        self.assertFalse(ev.unhealthy(pose))
        g = float(ev.ground(-20.0, 0.0))
        up = pose.copy()
        up[0], up[2] = -20.0, pose[2] + g                    # same clearance, 20 m up the slope
        self.assertFalse(ev.unhealthy(up))
        down = up.copy()
        down[2] = g                                           # torso origin on the slope surface -> touches
        self.assertTrue(ev.unhealthy(down))
        high = up.copy()
        high[0], high[2] = -38.0, float(ev.ground(-38.0, 0.0)) + pose[2]   # > 5 m above z = 0, but not above ground
        self.assertGreater(high[2], 5.0)
        self.assertFalse(ev.unhealthy(high))
        ev.ground = None                                      # old behaviour: absolute height
        self.assertTrue(ev.unhealthy(high))
        ev.close()

    def test_vectorised(self):
        g = te.ground_height("rough_0.06")
        x = np.full((4, 5), -10.0)
        y = np.zeros((4, 5))
        self.assertEqual(g(x, y).shape, (4, 5))
        self.assertIsNone(te.ground_height("rocks_0.1"))
        self.assertIsNone(te.ground_height("ice_0.3"))

    def test_short_episode_runs(self):
        cfg = MPPIConfig(threads=2, episode_control_steps=3, warmup_samples=64)
        for t in ("slope_10", "rough_0.06", "rocks_0.1", "ice_0.3", "ridged_s0.75"):
            r = evaluate_design(QUAD, t, 1, 0, cfg)
            self.assertEqual(r["episodes"][0]["env_steps"], 12, t)
            self.assertTrue(math.isfinite(r["fitness"]))


class TextTests(unittest.TestCase):
    def test_beams_text_unchanged(self):
        self.assertEqual(TERRAIN["beams_s1"], TERRAIN["beams"])
        self.assertIn("beams", TERRAIN)
        self.assertIn("slope_10", TERRAIN)
        self.assertNotIn("slope", TERRAIN)

    def test_descriptions(self):
        expect = {"ridged_s0.75": ["0.15 m", "1.5 m", "1 m ahead"], "beams_s1.25": ["0.25 m", "0.375 m", "2.5 m"],
                  "ice_0.3": ["0.3"], "slope_10": ["10 degrees"], "rocks_0.14": ["0.07", "0.14 m"],
                  "rough_0.06": ["0.06 m"]}
        advice = ("should", "try ", "recommend", "keep the body", "stay low", "keep low", "crouch")
        for t, words in expect.items():
            text = TERRAIN[t]
            for w in words:
                self.assertIn(w, text, (t, w))
            for w in advice:
                self.assertNotIn(w, text.lower(), (t, w))
            self.assertTrue(NAME[t].startswith("the "))

    def test_prompts_adapt(self):
        for t in NEW:
            for text in (llm_mutation.system_prompt(8, True, t), selection.system_prompt(8, t)):
                self.assertIn("### TERRAIN ###\n" + TERRAIN[t], text)
                task = text[text.index("### TASK ###"):]
                self.assertNotIn("flat ground", task)
                self.assertIn(NAME[t], text)
            self.assertNotEqual(llm_mutation.prompt_sha256(8, True, t), llm_mutation.prompt_sha256(8, True, "beams"))


class ScriptTests(unittest.TestCase):
    def test_calibration_and_choice(self):
        from hyblock import calib_terrain
        tmp = Path(tempfile.mkdtemp())
        try:
            flat = tmp / "flat.csv"
            with open(flat, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["name", "design", "modules", "distance_m", "ep_scores"])
                w.writeheader()
                w.writerow({"name": "rand_000", "design": HAND["hand_quad"], "modules": 5, "distance_m": 3.0,
                            "ep_scores": json.dumps([70.0, 50.0, 60.0])})
                w.writerow({"name": "rand_001", "design": HAND["hand_inchworm"], "modules": 3, "distance_m": 0.2,
                            "ep_scores": json.dumps([4.0, 4.0, 4.0])})
            calib_terrain.main(["--flat", str(flat), "--out", str(tmp / "c"), "--terrains", "slope_5,ridged_s0.6",
                                "--episodes", "1", "--steps", "2", "--threads", "1"])
            with open(tmp / "c" / "terrain.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(sorted(r["terrain"] for r in rows), ["ridged_s0.6", "slope_5"])   # only the walker
            calib_terrain.main(["--flat", str(flat), "--out", str(tmp / "c"), "--terrains", "slope_5,ridged_s0.6",
                                "--episodes", "1", "--steps", "2", "--threads", "1"])                # resume: no dup
            self.assertEqual(len(list(csv.DictReader(open(tmp / "c" / "terrain.csv", encoding="utf-8")))), 2)
            ch = json.loads((tmp / "c" / "chosen.json").read_text(encoding="utf-8"))
            self.assertEqual(set(ch), {"slope", "ridged"})
            calib_terrain.main(["--flat", str(flat), "--out", str(tmp / "c"), "--terrains", "slope_5,ridged_s0.6",
                                "--episodes", "1", "--steps", "2", "--threads", "1", "--auto_insert"])
            ins = (tmp / "c" / "inserted.txt").read_text(encoding="utf-8").split()
            self.assertEqual(sorted(ins), ["ridged_s0.45", "slope_2.5"])                     # both 0 / 1 -> easier
            ch = json.loads((tmp / "c" / "chosen.json").read_text(encoding="utf-8"))
            self.assertEqual({c["status"] for c in ch.values()}, {"dropped"})
        finally:
            shutil.rmtree(tmp)

    def test_choice_rule(self):
        from hyblock.calib_terrain import choose
        terr = ["slope_5", "slope_10", "slope_15", "ice_0.5", "ice_0.3", "ice_0.2", "rocks_0.06", "rocks_0.1",
                "rocks_0.14"]
        rate = {"slope_5": (40, 120, 75), "slope_10": (30, 120, 75), "slope_15": (25, 120, 75),   # 0.33 0.25 0.21
                "ice_0.5": (36, 120, 75), "ice_0.3": (30, 120, 75), "ice_0.2": (26, 120, 75),     # 0.30 0.25 0.22
                "rocks_0.06": (60, 120, 75), "rocks_0.1": (50, 120, 75), "rocks_0.14": (40, 120, 75)}
        ch = choose(rate, terr, 75)
        self.assertEqual(ch["slope"]["terrain"], "slope_15")       # hardest in range
        self.assertEqual(ch["ice"]["terrain"], "ice_0.2")          # smaller friction = harder
        self.assertEqual(ch["rocks"]["status"], "too_easy")
        self.assertEqual(ch["slope"]["fallback"], "slope_10")
        self.assertEqual(ch["rocks"]["insert"], "rocks_0.2")
        rate.update({"ice_0.5": (20, 120, 75), "ice_0.3": (10, 120, 75), "ice_0.2": (5, 120, 75)})
        self.assertEqual(choose(rate, terr, 75)["ice"]["insert"], "ice_0.8")               # all too hard -> easier
        rate.update({"slope_5": (45, 120, 75), "slope_10": (40, 120, 75), "slope_15": (10, 120, 75)})
        self.assertEqual(choose(rate, terr, 75)["slope"]["insert"], "slope_12.5")          # straddles -> midpoint
        self.assertEqual(choose(rate, terr, 75, final=True)["slope"]["status"], "dropped")
        self.assertEqual(choose({"beams_s1.25": (29, 120, 75)}, ["beams_s1.25"], 75)["beams"]["fallback"], "beams_s1.5")
        rate["slope_15"] = (25, 120, 70)
        self.assertEqual(choose(rate, terr, 75)["slope"]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def _run(d: Path, g0: float, late: float):
        d.mkdir(parents=True)
        cols = ["gen", "idx", "status", "fitness", "mpc_s", "llm_in", "llm_out", "sel_slot", "sel_pred", "modules"]
        with open(d / "evals.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for gen in range(8):
                for i in range(16):
                    w.writerow({"gen": gen, "idx": i, "status": "mpc", "fitness": g0 if gen < 5 else late + i,
                                "mpc_s": 1, "llm_in": 0, "llm_out": 0, "sel_slot": "", "sel_pred": "", "modules": 4})
        (d / "run_meta.json").write_text("{}", encoding="utf-8")
        (d / "DONE").write_text("", encoding="utf-8")

    def test_pilot_and_formal(self):
        from hyblock import terrain_ext_analysis as ta
        tmp = Path(tempfile.mkdtemp())
        try:
            for s in (0, 1, 2):
                self._run(tmp / "pilot" / "slope_10" / f"ga_s{s}", 10.0, 20.0)     # late median 27.5 < 30 -> fail
                self._run(tmp / "pilot" / "rocks_0.1" / f"ga_s{s}", 5.0, 40.0)     # pass
            (tmp / "pilot" / "analysis").mkdir()
            ta.main(["pilot", "--root", str(tmp / "pilot"), "--out", str(tmp / "pilot" / "analysis")])
            v = json.loads((tmp / "pilot" / "analysis" / "pilot_verdict.json").read_text(encoding="utf-8"))
            self.assertEqual(v, {"rocks_0.1": "pass", "slope_10": "fail"})
            for s in range(500, 504):
                self._run(tmp / "formal" / "rocks_0.1" / f"ga_s{s}", 5.0, 40.0)
                self._run(tmp / "formal" / "rocks_0.1" / f"ga_llmtyped_sel_llmEv_s{s}", 5.0, 50.0)
            ta.main(["formal", "--root", str(tmp / "formal"), "--seeds", "500-503", "--out", str(tmp / "fa")])
            rep = (tmp / "fa" / "report.md").read_text(encoding="utf-8")
            self.assertIn("| B6 − B1 后期中位数 | 4 | +10.0 |", rep)
            self.assertIn("1 / 1", rep)
        finally:
            shutil.rmtree(tmp)
