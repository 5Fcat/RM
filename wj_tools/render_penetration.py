"""Render the initial pose of a Hyblock design and paint the interpenetrating parts red.
usage: python render_penetration.py SRC "design text" out.png [--az -120 --el -25 --dist 1.6]"""
import argparse, re, sys
p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("design"); p.add_argument("out")
p.add_argument("--az", type=float, default=-120); p.add_argument("--el", type=float, default=-25)
p.add_argument("--dist", type=float, default=1.6); p.add_argument("--w", type=int, default=1200); p.add_argument("--h", type=int, default=900)
a = p.parse_args()
sys.path.insert(0, a.src)
import mujoco
import numpy as np
from PIL import Image
from hyblock.mjcf import compile_design

xml = compile_design(a.design, max_modules=10, check_self_collision=False, self_collision=True)
m0 = mujoco.MjModel.from_xml_string(xml)
d0 = mujoco.MjData(m0)
mujoco.mj_forward(m0, d0)
bad = set()
for i in range(d0.ncon):
    c = d0.contact[i]
    if c.dist < -0.005:
        n1, n2 = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if n1 and n2 and "floor" not in (n1 + n2):
            bad |= {n1, n2}
print("penetrating geoms:", sorted(bad))

# look: same as render_video.py (light floor, slide colours), penetrating parts red
xml = re.sub(r'<texture name="grid"[^>]*/>',
             '<texture name="grid" type="2d" builtin="checker" mark="edge" width="512" height="512" '
             'rgb1=".97 .97 .98" rgb2=".97 .97 .98" markrgb=".80 .82 .86" />', xml)
xml = xml.replace('texrepeat="1 1" texuniform="true" reflectance=".2"', 'texrepeat="2 2" texuniform="true" reflectance="0" emission=".5"')
xml = re.sub(r'<light directional="true"[^>]*/>', '<light directional="true" castshadow="false" diffuse=".7 .7 .7" specular=".05 .05 .05" dir="-.25 .25 -1" />', xml)
xml = xml.replace("<asset>", '<asset>\n    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".93 .95 .98" width="256" height="256" />', 1)
xml = xml.replace("<worldbody>", '<visual><headlight ambient=".45 .45 .45" diffuse=".3 .3 .3" specular="0 0 0"/>'
                  '<quality offsamples="8"/><global offwidth="1920" offheight="1440"/></visual>\n  <worldbody>', 1)
COL = {"foot": ".15 .15 .17 1", "block": ".86 .86 .88 1", "P1": ".12 .18 .55 1", "P2": ".12 .18 .55 1", "hub": ".45 .55 .72 1"}
RED = ".80 .12 .18 1"
def recol(mt):
    name, rest = mt.group(1), mt.group(2)
    if name in bad:
        return f'<geom name="{name}"' + re.sub(r'rgba="[^"]+"', f'rgba="{RED}"', rest)
    key = next((k for k in COL if f"_{k}" in name), None)
    if key is None:
        return mt.group(0)
    return f'<geom name="{name}"' + re.sub(r'rgba="[^"]+"', f'rgba="{COL[key]}"', rest)
xml = re.sub(r'<geom name="([^"]+)"([^>]*>)', recol, xml)
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, a.h, a.w)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.azimuth, cam.elevation, cam.distance = a.az, a.el, a.dist
cam.lookat[:] = [d.subtree_com[1][0], d.subtree_com[1][1], d.subtree_com[1][2] * 0.8]
r.update_scene(d, cam)
Image.fromarray(r.render()).save(a.out)
print("wrote", a.out)
