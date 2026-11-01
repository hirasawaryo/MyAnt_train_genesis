"""Parametric Ant (MJCF) generator.

Geometry follows nv_ant.xml (the source of Isaac Lab's Ant / MyAnt.usda):
torso sphere r=0.25, hip (aux) capsule (0.2, 0.2), leg capsule and foot capsule
along the diagonal, all capsules r=0.08, density 5, motor gear 15.

x1 = leg_length  (default 0.2828427255153656 = |(0.2, 0.2)|)
x2 = foot_length (default 0.5656854510307312 = |(0.4, 0.4)|)
"""

from __future__ import annotations

import math
from pathlib import Path

DEFAULT_LEG_LENGTH = 0.2828427255153656
DEFAULT_FOOT_LENGTH = 0.5656854510307312

# (name, sign_x, sign_y, ankle_axis, ankle_range)
_LEGS = [
    ("front_left", 1, 1, "-1 1 0", "30 100"),
    ("front_right", -1, 1, "1 1 0", "-100 -30"),
    ("left_back", -1, -1, "-1 1 0", "-100 -30"),
    ("right_back", 1, -1, "1 1 0", "30 100"),
]

# initial foot joint angles (rad), same as Isaac Lab ANT_CFG
INIT_FOOT_ANGLE = {
    "front_left_foot": 0.785398,
    "front_right_foot": -0.785398,
    "left_back_foot": -0.785398,
    "right_back_foot": 0.785398,
}


def spawn_height(foot_length: float) -> float:
    """Torso height so the 45-degree bent feet start just above the ground."""
    return max(0.75, foot_length * math.sin(math.pi / 4) + 0.25)


def make_ant_xml(leg_length: float, foot_length: float) -> str:
    d_leg = leg_length / math.sqrt(2)
    d_foot = foot_length / math.sqrt(2)
    bodies = []
    for name, sx, sy, ankle_axis, ankle_range in _LEGS:
        bodies.append(
            f"""
      <geom fromto="0 0 0 {0.2 * sx} {0.2 * sy} 0" name="{name}_aux_geom" size="0.08" type="capsule"/>
      <body name="{name}_leg" pos="{0.2 * sx} {0.2 * sy} 0">
        <joint axis="0 0 1" name="{name}_leg" pos="0 0 0" range="-40 40" type="hinge"/>
        <geom fromto="0 0 0 {d_leg * sx} {d_leg * sy} 0" name="{name}_leg_geom" size="0.08" type="capsule"/>
        <body name="{name}_foot" pos="{d_leg * sx} {d_leg * sy} 0">
          <joint axis="{ankle_axis}" name="{name}_foot" pos="0 0 0" range="{ankle_range}" type="hinge"/>
          <geom fromto="0 0 0 {d_foot * sx} {d_foot * sy} 0" name="{name}_foot_geom" size="0.08" type="capsule"/>
        </body>
      </body>"""
        )
    actuators = "\n".join(
        f'    <motor ctrllimited="true" ctrlrange="-1 1" joint="{n}_{p}" gear="15"/>'
        for p in ("leg", "foot")
        for n, *_ in _LEGS
    )
    return f"""<mujoco model="myant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <default>
    <joint armature="0.01" damping="0.1" limited="true"/>
    <geom contype="1" conaffinity="1" condim="3" density="5.0" friction="1.0 0.5 0.5" rgba="0.97 0.38 0.06 1"/>
  </default>
  <worldbody>
    <body name="torso" pos="0 0 {spawn_height(foot_length)}">
      <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
      <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" type="free"/>{''.join(bodies)}
    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


def write_ant_xml(leg_length: float, foot_length: float, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_ant_xml(leg_length, foot_length))
    return path
