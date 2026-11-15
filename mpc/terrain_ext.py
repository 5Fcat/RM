"""Extra terrains for the Hyblock domain (11-13, 地形扩展计划 = 正式实验计划 变更记录第 22 条).

Terrain names carry their difficulty parameter, so the LLM prompt (hyblock/terrain_text.py) can state the
exact geometry and `--terrain_scale` stays 1.0:

    ridged_s<scale>   RoboMorph ridges (half-buried cylinders), radius and spacing x scale  e.g. ridged_s0.75
                      (first ridge at max(1, scale) m so that it never overlaps the robot at reset; as many
                       ridges as fit in 40 m)
    beams_s<scale>    RoboMorph beams with all lengths x scale (scale >= 1; built by mpc.mppi) e.g. beams_s1.25
    ice_<mu>          whole floor with friction mu (the floor has priority 1, so mu sets the contact friction)
    slope_<deg>       uphill ramp of <deg> degrees starting 1 m ahead
    rocks_<h>         scattered box-shaped rocks, heights in [h/2, h] m, starting 1.25 m ahead
    rough_<a>         smooth uneven ground (height field), heights in [0, a] m, starting 1 m ahead

The robot walks in -x. Everything within 1 m of the start (x > -1) is flat floor, so the initial pose and the
① screen are the same as on flat ground. Random layouts use a fixed terrain seed: every design, episode and run
sees the same terrain. The old names (flat, ridged, beams, frozen_lake) never reach this module, so their XML is
unchanged byte for byte.

`ground_height(name)` returns z = g(x, y) of the terrain surface (None when the ground is the z = 0 floor). The
MPC evaluator measures torso height relative to it, so "torso touches the ground" and the 5 m upper bound mean
the same thing on a slope as on flat ground.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

import numpy as np

START_M = 1.0            # terrain area starts this far ahead of the start position (= first beam / ridge)
LENGTH_M = 40.0          # length of the terrain area in the direction of travel
HALF_WIDTH_M = 4.0       # rocks / rough ground cover |y| <= 4 m (slope: |y| <= 10 m, like the beams)
TERRAIN_SEED = 20261113
ROCK_DENSITY = 1.0       # rocks per m^2
ROCK_HALF_XY = (0.06, 0.15)
ROCK_START_M = 1.25      # rock centres at least this far ahead (nearest rock edge >= ~1.04 m ahead)
ROUGH_RES_M = 0.1        # height-field grid spacing
ROUGH_WAVES = 12
ROUGH_WAVELENGTH = (0.8, 2.0)
ROUGH_RAMP_M = 1.0       # the uneven ground rises from 0 to full amplitude over its first metre
ROUGH_BASE_M = 0.05
SLOPE_HALF_THICK = 0.1

_PAT = re.compile(r"^(ridged_s|beams_s|ice_|slope_|rocks_|rough_)(\d+(?:\.\d+)?)$")
KINDS = ("ridged", "beams", "ice", "slope", "rocks", "rough")
CONTINUOUS = ("ice", "slope", "rocks", "rough")     # no discrete first obstacle (pass = progress, see pass_need)


def parse(name: str):
    """(kind, value) for an extended terrain name, None for anything else (flat, ridged, beams, ...)."""
    m = _PAT.match(name)
    if not m:
        return None
    kind = m.group(1).rstrip("_").removesuffix("_s")
    v = float(m.group(2))
    if v <= 0 or (kind == "slope" and v >= 45) or (kind == "ice" and v > 2) or (kind == "beams" and v < 1):
        raise ValueError(f"terrain parameter out of range: {name}")
    return kind, v


def is_extended(name: str) -> bool:
    return parse(name) is not None


def pass_need(name: str) -> float:
    """Displacement in -x [m] that counts as "passing" in the calibration (before the x_start correction).

    ridged / beams: the torso origin 0.5 m past the far edge of the first obstacle (as hyblock/calibrate.py).
    continuous terrains: 2 m into the terrain area, i.e. 3 m from the start (ice: the whole floor, same 3 m)."""
    kind, v = parse(name)
    if kind == "ridged":
        return ridge_first(v) + 0.2 * v + 0.5
    if kind == "beams":
        return (1.0 + 0.2) * v + 0.5
    return START_M + 2.0


def ridge_first(s: float) -> float:
    return max(START_M, 1.0 * s)


# ----------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------

def _floor(root):
    floors = root.findall(".//geom[@type='plane']")
    if len(floors) != 1:
        raise ValueError(f"expected exactly 1 ground plane, found {len(floors)}")
    return floors[0]


def _surface_attrs(floor) -> dict:
    """Terrain geoms copy the floor's contact settings, so the floor/terrain sets the friction everywhere."""
    out = {"contype": "0", "conaffinity": "1"}
    for k in ("priority", "friction"):
        if floor.get(k) is not None:
            out[k] = floor.get(k)
    return out


def rocks_layout(h: float) -> np.ndarray:
    """(n, 6): x, y, half_x, half_y, height, yaw. Same layout for every h (heights scale with h)."""
    rng = np.random.default_rng(TERRAIN_SEED)
    area = (START_M + LENGTH_M - ROCK_START_M) * 2 * HALF_WIDTH_M
    n = int(round(ROCK_DENSITY * area))
    x = -rng.uniform(ROCK_START_M, START_M + LENGTH_M, n)
    y = rng.uniform(-HALF_WIDTH_M, HALF_WIDTH_M, n)
    hx = rng.uniform(*ROCK_HALF_XY, n)
    hy = rng.uniform(*ROCK_HALF_XY, n)
    frac = rng.uniform(0.5, 1.0, n)
    yaw = rng.uniform(0, math.pi, n)
    return np.column_stack([x, y, hx, hy, frac * h, yaw])


def rough_grid(a: float):
    """Height field on a regular grid: (xs, ys, heights[ny, nx]) with heights in [0, a], max exactly a."""
    rng = np.random.default_rng(TERRAIN_SEED + 1)
    xs = -START_M - np.arange(0.0, LENGTH_M + 1e-9, ROUGH_RES_M)      # 0 m ... 40 m into the area (-x)
    ys = np.arange(-HALF_WIDTH_M, HALF_WIDTH_M + 1e-9, ROUGH_RES_M)
    X, Y = np.meshgrid(xs, ys)
    f = np.zeros_like(X)
    for _ in range(ROUGH_WAVES):
        lam = rng.uniform(*ROUGH_WAVELENGTH)
        phi = rng.uniform(0, 2 * math.pi)
        psi = rng.uniform(0, 2 * math.pi)
        f += np.sin(2 * math.pi * (X * math.cos(phi) + Y * math.sin(phi)) / lam + psi)
    n = (f - f.min()) / (f.max() - f.min())
    n *= np.clip((-START_M - X) / ROUGH_RAMP_M, 0.0, 1.0)
    n /= n.max()
    return xs, ys, a * n


def add_terrain(xml_string: str, name: str) -> str:
    kind, v = parse(name)
    if kind == "beams":
        raise ValueError("beams_s is built by mpc.mppi.apply_terrain")
    root = ET.fromstring(xml_string)
    wb = root.find(".//worldbody")
    floor = _floor(root)
    attrs = _surface_attrs(floor)
    rgba = "0.6 0.55 0.45 1.0"
    if kind == "ridged":
        spread, radius, first = 2.0 * v, 0.2 * v, ridge_first(v)
        n = int(math.floor(LENGTH_M / spread)) + 1
        for i in range(n):
            ET.SubElement(wb, "geom", name=f"ridge_{i}", type="cylinder", size=f"{radius:.6g} 10.0",
                          pos=f"-{i * spread + first:.6g} 0 0",
                          quat="0.7071067811865476 0.7071067811865476 0.0 0.0", rgba="0.7 0.5 0.3 1.0", **attrs)
    elif kind == "ice":
        fr = (floor.get("friction") or "1 0.005 0.0001").split()
        floor.set("friction", " ".join([f"{v:.6g}"] + fr[1:]))
    elif kind == "slope":
        th = math.radians(v)
        L = START_M + LENGTH_M
        u = np.array([-math.cos(th), 0.0, math.sin(th)])            # up-slope direction (-x, rising)
        nrm = np.array([math.sin(th), 0.0, math.cos(th)])           # surface normal
        c = np.array([-START_M, 0.0, 0.0]) + (L / 2) * u - SLOPE_HALF_THICK * nrm
        ET.SubElement(wb, "geom", name="slope", type="box", size=f"{L / 2:.6g} 10 {SLOPE_HALF_THICK:.6g}",
                      pos=f"{c[0]:.9g} {c[1]:.9g} {c[2]:.9g}",
                      quat=f"{math.cos(th / 2):.12g} 0 {math.sin(th / 2):.12g} 0", rgba=rgba, **attrs)
    elif kind == "rocks":
        for i, (x, y, hx, hy, hz, yaw) in enumerate(rocks_layout(v)):
            ET.SubElement(wb, "geom", name=f"rock_{i}", type="box", size=f"{hx:.6g} {hy:.6g} {hz / 2:.6g}",
                          pos=f"{x:.6g} {y:.6g} {hz / 2:.6g}",
                          quat=f"{math.cos(yaw / 2):.9g} 0 0 {math.sin(yaw / 2):.9g}", rgba=rgba, **attrs)
    elif kind == "rough":
        xs, ys, H = rough_grid(v)
        # MuJoCo height field: columns run along +x, rows along +y, and the `elevation` attribute is read
        # top row first (rows reversed). Our xs run along -x -> flip columns; rows flipped for the attribute.
        data = (H / v)[::-1, ::-1]
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        hx_, hy_ = (len(xs) - 1) * ROUGH_RES_M / 2, (len(ys) - 1) * ROUGH_RES_M / 2
        ET.SubElement(asset, "hfield", name="rough", nrow=str(H.shape[0]), ncol=str(H.shape[1]),
                      size=f"{hx_:.6g} {hy_:.6g} {v:.6g} {ROUGH_BASE_M:.6g}",
                      elevation=" ".join(f"{z:.5f}" for z in data.ravel()))
        ET.SubElement(wb, "geom", name="rough", type="hfield", hfield="rough",
                      pos=f"{-START_M - hx_:.6g} 0 0", rgba=rgba, **attrs)
    return ET.tostring(root, encoding="unicode")


# ----------------------------------------------------------------------------------------------
# surface height for the termination test
# ----------------------------------------------------------------------------------------------

def ground_height(name: str):
    """Vectorised g(x, y) of the walking surface, or None if it is the z = 0 floor (rocks are obstacles)."""
    p = parse(name)
    if p is None:
        return None
    kind, v = p
    if kind == "slope":
        t = math.tan(math.radians(v))
        L = START_M + LENGTH_M

        def g(x, y):
            return np.clip(-START_M - np.asarray(x, float), 0.0, L * math.cos(math.radians(v))) * t
        return g
    if kind == "rough":
        xs, ys, H = rough_grid(v)
        x0, y0 = xs[0], ys[0]

        def g(x, y):
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            fx = (x0 - x) / ROUGH_RES_M                     # xs decrease along the columns
            fy = (y - y0) / ROUGH_RES_M
            inside = (fx >= 0) & (fx <= len(xs) - 1) & (fy >= 0) & (fy <= len(ys) - 1)
            fx = np.clip(fx, 0, len(xs) - 1.000001)
            fy = np.clip(fy, 0, len(ys) - 1.000001)
            i, j = fy.astype(int), fx.astype(int)
            ty, tx = fy - i, fx - j
            z = (H[i, j] * (1 - tx) * (1 - ty) + H[i, j + 1] * tx * (1 - ty)
                 + H[i + 1, j] * (1 - tx) * ty + H[i + 1, j + 1] * tx * ty)
            return np.where(inside, z, 0.0)
        return g
    return None
