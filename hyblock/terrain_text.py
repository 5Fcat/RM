"""LLM 提示词里的地形描述（第 12 周，横梁）。

平地（flat）：原样返回，提示词逐字节不变（prompt_sha256 与 LLM 缓存键不变）。
其他地形：把任务说明里写死的 "flat ground" 按固定规则替换，并在 "### TASK ###" 前加一段 ### TERRAIN ###。
地形段只写事实（几何尺寸、位置、能否绕开），不写从搜索结果得出的建议（第12周记录 第 2a 节）。
"""
from __future__ import annotations

TERRAIN = {
    "beams": (
        "The robot moves over flat ground (the same high-friction floor as on flat terrain) with horizontal "
        "cylindrical beams fixed across its path. Each beam has a radius of 0.2 m and its axis is 0.5 m above the "
        "ground, so the lower edge of a beam is 0.3 m and its top is 0.7 m above the ground. The first beam is 1 m "
        "ahead of the robot's starting position and the next beams follow every 2 m in the direction of travel. "
        "The beams span the whole width of the terrain, so the robot cannot go around them, and they do not move."
    ),
}
NAME = {"beams": "the beams terrain"}


# ---- 11-13 extended terrains (mpc/terrain_ext.py; the name carries the difficulty) -----------------------
# Facts only (geometry, friction, where it starts, whether it can be avoided); no advice. beams_s1 gives
# exactly the frozen "beams" text above (tests/test_terrains.py checks it).

def _n(x: float) -> str:
    return f"{round(x, 6):g}"


_BEAMS_T = ("The robot moves over flat ground (the same high-friction floor as on flat terrain) with horizontal "
            "cylindrical beams fixed across its path. Each beam has a radius of {r} m and its axis is {h} m above the "
            "ground, so the lower edge of a beam is {lo} m and its top is {hi} m above the ground. The first beam is "
            "{first} m ahead of the robot's starting position and the next beams follow every {gap} m in the direction "
            "of travel. The beams span the whole width of the terrain, so the robot cannot go around them, and they "
            "do not move.")


def _describe(kind: str, v: float) -> str:
    if kind == "beams":
        return _BEAMS_T.format(r=_n(0.2 * v), h=_n(0.5 * v), lo=_n(0.3 * v), hi=_n(0.7 * v), first=_n(1.0 * v),
                               gap=_n(2.0 * v))
    if kind == "ridged":
        return ("The robot moves over flat ground (the same high-friction floor as on flat terrain) crossed by low "
                f"ridges. Each ridge is a horizontal cylinder of radius {_n(0.2 * v)} m lying half buried in the "
                f"ground, so it forms a rounded bump {_n(0.2 * v)} m high and {_n(0.4 * v)} m wide. The first ridge "
                f"is {_n(max(1.0, v))} m ahead of the robot's starting position and the next ridges follow every "
                f"{_n(2.0 * v)} m in the direction of travel. The ridges span the whole width of the terrain, so the "
                "robot cannot go around them, and they do not move.")
    if kind == "ice":
        return ("The robot moves over flat ground covered with ice: the whole floor is slippery, with a friction "
                f"coefficient of {_n(v)} (the normal floor on flat terrain has 1.0). There are no obstacles.")
    if kind == "slope":
        return ("The robot starts on flat ground. 1 m ahead of its starting position the ground turns into a "
                f"straight uphill slope of {_n(v)} degrees that rises in the direction of travel and continues for "
                "40 m. The slope has the same high-friction surface as the flat floor and spans the whole width of "
                "the terrain.")
    if kind == "rocks":
        return ("The robot moves over flat ground (the same high-friction floor as on flat terrain) with box-shaped "
                "rocks scattered on it at random, about one rock per square metre, so there are gaps between them. "
                f"Each rock is 0.12-0.3 m long and wide and between {_n(v / 2)} and {_n(v)} m high. The rocks are "
                "fixed and do not move. The first rocks are about 1 m ahead of the robot's starting position and the "
                "rocks continue for 40 m in the direction of travel.")
    if kind == "rough":
        return ("The robot starts on flat ground. From 1 m ahead of its starting position the ground becomes "
                f"uneven: a smooth, wavy surface whose height varies between 0 and {_n(v)} m, with bumps and dips "
                "roughly 0.8-2 m apart. It has the same high-friction surface as the flat floor and continues for "
                "40 m in the direction of travel.")
    raise ValueError(kind)


_TITLE = {"ridged": "the ridged terrain", "beams": "the beams terrain", "ice": "the icy terrain",
          "slope": "the slope terrain", "rocks": "the rocky terrain", "rough": "the uneven terrain"}


class _Lookup(dict):
    """TERRAIN / NAME: fixed entries as before; extended names (e.g. slope_10) are generated on lookup."""

    def __init__(self, base, make):
        super().__init__(base)
        self._make = make

    def __missing__(self, key):
        from mpc.terrain_ext import parse
        p = parse(key)
        if p is None:
            raise KeyError(key)
        return self._make(*p)

    def __contains__(self, key):
        if dict.__contains__(self, key):
            return True
        from mpc.terrain_ext import parse
        return parse(key) is not None


TERRAIN = _Lookup(TERRAIN, _describe)
NAME = _Lookup(NAME, lambda kind, v: _TITLE[kind])

# (原文, 替换) —— 各提示词里出现的写法；每个地形版提示词至少要替换掉一处，且替换后不能再有 "flat ground"
RULES = [
    ("robots that walk fast on flat ground", "robots that move forward fast on {name}"),
    ("simulated on flat ground for", "simulated on {name} for"),
    ("walk fast on flat ground", "move forward fast on {name}"),
]
TASK_MARK = "### TASK ###"


def adapt(text: str, terrain: str = "flat") -> str:
    if terrain == "flat":
        return text
    if terrain not in TERRAIN:
        raise ValueError(f"no terrain description for {terrain!r} (hyblock/terrain_text.py)")
    n = 0
    for old, new in RULES:
        n += text.count(old)
        text = text.replace(old, new.format(name=NAME[terrain]))
    if n == 0 or "flat ground" in text:
        raise ValueError(f"terrain_text.adapt: prompt does not match the replacement rules ({n} replaced)")
    if text.count(TASK_MARK) != 1:
        raise ValueError("terrain_text.adapt: expected exactly one '### TASK ###'")
    section = f"### TERRAIN ###\n{TERRAIN[terrain]}\n\n"
    return text.replace(TASK_MARK, section + TASK_MARK)
