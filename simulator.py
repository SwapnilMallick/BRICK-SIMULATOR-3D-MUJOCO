from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import mujoco


WORKSPACE_ROOT = Path(__file__).resolve().parent
BLENDER_DIR = WORKSPACE_ROOT / "BlenderFiles"
BUILD_DIR = WORKSPACE_ROOT / "build"


@dataclass(frozen=True)
class BrickType:
    group_name: str
    prefix: str
    display_name: str
    size_xyz: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    mesh_stem: str | None = None
    add_lego_studs: bool = False


@dataclass(frozen=True)
class InventoryLayout:
    rows: int = 5
    columns: int = 5
    layers: int = 10
    gap_x: float = 0.75
    gap_y: float = 0.1
    gap_z: float = 0.75
    corner_inset: float = 1.0
    separation_gap: float = 5.0


@dataclass(frozen=True)
class WorldBounds:
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    length: float = 101.0
    width: float = 101.0
    wall_height: float = 3.0
    wall_thickness: float = 0.5
    floor_thickness: float = 0.2


# NORMAL_BRICK = BrickType(
#     group_name="Inventory_NormalBricks",
#     prefix="InventoryBrick_",
#     display_name="normal_brick",
#     size_xyz=(5.0, 2.0, 2.0),
#     rgba=(0.73, 0.73, 0.75, 1.0),
# )

# PROCEDURAL_LEGO_BRICK = BrickType(
#     group_name="Inventory_LegoBricks",
#     prefix="InventoryLegoBrick_",
#     display_name="procedural_lego_brick",
#     size_xyz=(4.0, 0.96, 1.6),
#     rgba=(0.88, 0.15, 0.15, 1.0),
#     add_lego_studs=True,
# )

# ORANGE_LEGO_BRICK = BrickType(
#     group_name="Inventory_OrangeLegoBricks",
#     prefix="InventoryOrangeLegoBrick_",
#     display_name="orange_lego_brick",
#     size_xyz=(4.0, 0.96, 1.6),
#     rgba=(0.95, 0.45, 0.12, 1.0),
#     mesh_stem="orangeLEGOBrick",
#     add_lego_studs=True,
# )

# GREEN_PLAEX_LONG_BRICK = BrickType(
#     group_name="Inventory_GreenPlaexLongBricks",
#     prefix="InventoryGreenPlaexLongBrick_",
#     display_name="green_plaex_long_brick",
#     size_xyz=(5.0, 2.0, 2.0),
#     rgba=(0.20, 0.62, 0.33, 1.0),
#     mesh_stem="greenPLAEXLong",
# )

#ORANGE_PLAEX_LONG_BRICK = BrickType(
#    group_name="Inventory_OrangePlaexLongBricks",
#    prefix="InventoryOrangePlaexLongBrick_",
#    display_name="orange_plaex_long_brick",
#    size_xyz=(5.0, 2.0, 2.0),
#    rgba=(0.91, 0.44, 0.17, 1.0),
#    mesh_stem="orangePLAEXLong",
#)

A2 = BrickType(
    group_name="Inventory_A2",
    prefix="InventoryA2_",
    display_name="a2_brick",
    size_xyz=(5.0, 2.0, 2.0),
    rgba=(0.93, 0.78, 0.20, 1.0),
    mesh_stem="yellowPLAEXLong",
)

C2 = BrickType(
    group_name="Inventory_C2",
    prefix="InventoryC2_",
    display_name="c2_brick",
    size_xyz=(2.0, 2.0, 2.0),
    rgba=(0.90, 0.83, 0.22, 1.0),
    mesh_stem="yellowPLAEXSide",
)

A1 = BrickType(
    group_name="Inventory_A1",
    prefix="InventoryA1_",
    display_name="a1_brick",
    size_xyz=(5.0, 1.0, 2.0),
    rgba=(0.85, 0.12, 0.12, 1.0),
    mesh_stem="redPLAEXLong",
)

C1 = BrickType(
    group_name="Inventory_C1",
    prefix="InventoryC1_",
    display_name="c1_brick",
    size_xyz=(2.0, 1.0, 2.0),
    rgba=(0.85, 0.12, 0.12, 1.0),
    mesh_stem="redPLAEXSide",
)

@dataclass(frozen=True)
class PickAction:
    brick_id: str


@dataclass(frozen=True)
class OrientAction:
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class PlaceAction:
    x: float
    y: float
    z: float


INVENTORY_ORDER: Sequence[BrickType] = (
    # NORMAL_BRICK,
    # PROCEDURAL_LEGO_BRICK,
    # ORANGE_LEGO_BRICK,
    # GREEN_PLAEX_LONG_BRICK,
    # ORANGE_PLAEX_LONG_BRICK,
    A2,
    C2,
    A1,
    C1,
)

# ORIGIN_PREVIEW_BRICK = GREEN_PLAEX_LONG_BRICK
# ORIGIN_PREVIEW_BRICK_ID = "PreviewGreenPlaexLongBrick_Origin"
# ORIGIN_PREVIEW_POSITION = (0.0, 1.0, 0.0)


def mujoco_mesh_path_for(stem: str | None) -> Path | None:
    if not stem:
        return None

    for suffix in (".obj", ".stl", ".msh"):
        candidate = BLENDER_DIR / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    return None


def compute_required_bounds(layout: InventoryLayout) -> WorldBounds:
    base = WorldBounds()

    widths_x = []
    depths_z = []
    for brick_type in INVENTORY_ORDER:
        sx, _, sz = brick_type.size_xyz
        widths_x.append((layout.columns * sx) + ((layout.columns - 1) * layout.gap_x))
        depths_z.append((layout.rows * sz) + ((layout.rows - 1) * layout.gap_z))

    required_length = (
        sum(widths_x)
        + ((len(INVENTORY_ORDER) - 1) * layout.separation_gap)
        + (2.0 * base.wall_thickness)
        + (2.0 * layout.corner_inset)
    )
    required_width = max(depths_z) + (2.0 * base.wall_thickness) + (2.0 * layout.corner_inset)

    return WorldBounds(
        center=base.center,
        length=max(base.length, required_length + 1.0),
        width=max(base.width, required_width + 1.0),
        wall_height=base.wall_height,
        wall_thickness=base.wall_thickness,
        floor_thickness=base.floor_thickness,
    )


def normal_inventory_start(bounds: WorldBounds, layout: InventoryLayout) -> tuple[float, float, float]:
    half_length = bounds.length * 0.5
    half_width = bounds.width * 0.5
    first_brick = INVENTORY_ORDER[0]
    start_x = bounds.center[0] - half_length + bounds.wall_thickness + (first_brick.size_xyz[0] * 0.5) + layout.corner_inset
    start_z = bounds.center[2] + half_width - bounds.wall_thickness - (first_brick.size_xyz[2] * 0.5) - layout.corner_inset
    return (start_x, 0.0, start_z)


def inventory_start_x(
    brick_type: BrickType,
    all_types: Sequence[BrickType],
    bounds: WorldBounds,
    layout: InventoryLayout,
) -> float:
    start_x, _, _ = normal_inventory_start(bounds, layout)
    current_x = start_x

    for i, current_type in enumerate(all_types):
        sx = current_type.size_xyz[0]
        if i == 0:
            type_start_x = current_x
        else:
            previous_type = all_types[i - 1]
            previous_width = ((layout.columns - 1) * (previous_type.size_xyz[0] + layout.gap_x)) + previous_type.size_xyz[0]
            previous_right_edge = current_x + previous_width - (previous_type.size_xyz[0] * 0.5)
            type_start_x = previous_right_edge + layout.separation_gap + (sx * 0.5)

        if current_type == brick_type:
            return type_start_x

        current_x = type_start_x

    raise ValueError(f"Brick type {brick_type.display_name} not found in inventory order.")


def iter_inventory_positions(
    brick_type: BrickType,
    bounds: WorldBounds,
    layout: InventoryLayout,
) -> Iterable[tuple[int, tuple[float, float, float]]]:
    start_x = inventory_start_x(brick_type, INVENTORY_ORDER, bounds, layout)
    _, _, start_z = normal_inventory_start(bounds, layout)
    sx, sy, sz = brick_type.size_xyz
    base_y = bounds.center[1] + (sy * 0.5)
    step_x = sx + layout.gap_x
    step_y = sy + layout.gap_y
    step_z = sz + layout.gap_z

    for layer in range(layout.layers):
        for row in range(layout.rows):
            for col in range(layout.columns):
                x = start_x + (col * step_x)
                y = base_y + (layer * step_y)
                z = start_z - (row * step_z)
                brick_index = (layer * layout.rows * layout.columns) + (row * layout.columns) + col + 1
                yield brick_index, (x, y, z)


def format_vec(values: Sequence[float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def unity_to_mujoco_position(position: Sequence[float]) -> tuple[float, float, float]:
    # Unity layout data in this project is Y-up; MuJoCo is Z-up.
    x, y, z = position
    return (x, z, y)


def unity_to_mujoco_size(size: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = size
    return (x, z, y)


def lego_stud_positions(size_xyz: tuple[float, float, float]) -> List[tuple[float, float, float]]:
    stud_pitch = 0.8
    stud_height = 0.18
    top_y = (size_xyz[1] * 0.5) + (stud_height * 0.5)
    start_x = -((5 - 1) * stud_pitch) * 0.5
    start_z = -((2 - 1) * stud_pitch) * 0.5
    studs = []
    for x_index in range(5):
        for z_index in range(2):
            studs.append((start_x + (x_index * stud_pitch), top_y, start_z + (z_index * stud_pitch)))
    return studs


def build_brick_body_xml(brick_type: BrickType, brick_id: str, position: tuple[float, float, float], static: bool = True) -> str:
    sx, sy, sz = brick_type.size_xyz
    half_extents = unity_to_mujoco_size((sx * 0.5, sy * 0.5, sz * 0.5))
    mesh_path = mujoco_mesh_path_for(brick_type.mesh_stem)
    euler_attr = ' euler="90 0 0"' if mesh_path is not None else ''
    body_lines = [
        f'    <body name="{brick_id}" pos="{format_vec(unity_to_mujoco_position(position))}"{euler_attr}>',
    ]
    if not static:
        body_lines.append('      <freejoint/>')

    if mesh_path is not None:
        mesh_name = f"{brick_type.display_name}_mesh"
        body_lines.append(
            f'      <geom name="{brick_id}_geom" type="mesh" mesh="{mesh_name}" rgba="{format_vec(brick_type.rgba)}" mass="1"/>'
        )
    else:
        body_lines.append(
            f'      <geom name="{brick_id}_geom" type="box" size="{format_vec(half_extents)}" rgba="{format_vec(brick_type.rgba)}" mass="1"/>'
        )
        if brick_type.add_lego_studs:
            stud_radius = 0.24
            stud_half_height = 0.09
            for stud_index, stud_pos in enumerate(lego_stud_positions(brick_type.size_xyz), start=1):
                body_lines.append(
                    f'      <geom name="{brick_id}_stud_{stud_index}" type="cylinder" pos="{format_vec(unity_to_mujoco_position(stud_pos))}" '
                    f'size="{stud_radius:.6f} {stud_half_height:.6f}" rgba="{format_vec(brick_type.rgba)}" mass="0.02"/>'
                )

    body_lines.append("    </body>")
    return "\n".join(body_lines)


def build_asset_section() -> str:
    mesh_lines = []
    seen_stems: set[str] = set()
    all_brick_types = list(INVENTORY_ORDER)  # + [ORIGIN_PREVIEW_BRICK]
    for brick_type in all_brick_types:
        if brick_type.mesh_stem is None or brick_type.mesh_stem in seen_stems:
            continue
        seen_stems.add(brick_type.mesh_stem)
        mesh_path = mujoco_mesh_path_for(brick_type.mesh_stem)
        if mesh_path is None:
            continue

        relative_path = mesh_path.relative_to(WORKSPACE_ROOT)
        mesh_lines.append(
            f'    <mesh name="{brick_type.display_name}_mesh" file="{relative_path.as_posix()}"/>'
        )

    if not mesh_lines:
        return "  <asset/>\n"

    return "  <asset>\n" + "\n".join(mesh_lines) + "\n  </asset>\n"


def build_worldbody_xml(bounds: WorldBounds, layout: InventoryLayout, dynamic_brick_id: str | None = None) -> str:
    half_length = bounds.length * 0.5
    half_width = bounds.width * 0.5
    wall_center_y = bounds.center[1] + (bounds.wall_height * 0.5)
    floor_half_extents = unity_to_mujoco_size((half_length, bounds.floor_thickness * 0.5, half_width))
    side_wall_half_extents = unity_to_mujoco_size((bounds.wall_thickness * 0.5, bounds.wall_height * 0.5, half_width))
    front_back_wall_half_extents = unity_to_mujoco_size((half_length, bounds.wall_height * 0.5, bounds.wall_thickness * 0.5))

    lines = [
        "  <worldbody>",
        '    <light name="key" pos="0 0 80" dir="0 0 -1" diffuse="0.9 0.9 0.9" specular="0.2 0.2 0.2"/>',
        '    <light name="fill" pos="0 -40 60" dir="0 0 -1" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>',
        '    <camera name="overview" pos="0 -160 120" xyaxes="1 0 0 0 0.6 0.8"/>',
        (
            '    <geom name="floor" type="box" '
            f'pos="{format_vec(unity_to_mujoco_position((bounds.center[0], bounds.center[1] - (bounds.floor_thickness * 0.5), bounds.center[2])))}" '
            f'size="{format_vec(floor_half_extents)}" '
            'rgba="0.82 0.84 0.88 1"/>'
        ),
        (
            '    <geom name="wall_left" type="box" '
            f'pos="{format_vec(unity_to_mujoco_position((bounds.center[0] - half_length + (bounds.wall_thickness * 0.5), wall_center_y, bounds.center[2])))}" '
            f'size="{format_vec(side_wall_half_extents)}" '
            'rgba="0.2 0.2 0.2 0.02"/>'
        ),
        (
            '    <geom name="wall_right" type="box" '
            f'pos="{format_vec(unity_to_mujoco_position((bounds.center[0] + half_length - (bounds.wall_thickness * 0.5), wall_center_y, bounds.center[2])))}" '
            f'size="{format_vec(side_wall_half_extents)}" '
            'rgba="0.2 0.2 0.2 0.02"/>'
        ),
        (
            '    <geom name="wall_front" type="box" '
            f'pos="{format_vec(unity_to_mujoco_position((bounds.center[0], wall_center_y, bounds.center[2] + half_width - (bounds.wall_thickness * 0.5))))}" '
            f'size="{format_vec(front_back_wall_half_extents)}" '
            'rgba="0.2 0.2 0.2 0.02"/>'
        ),
        (
            '    <geom name="wall_back" type="box" '
            f'pos="{format_vec(unity_to_mujoco_position((bounds.center[0], wall_center_y, bounds.center[2] - half_width + (bounds.wall_thickness * 0.5))))}" '
            f'size="{format_vec(front_back_wall_half_extents)}" '
            'rgba="0.2 0.2 0.2 0.02"/>'
        ),
        # build_brick_body_xml(ORIGIN_PREVIEW_BRICK, ORIGIN_PREVIEW_BRICK_ID, ORIGIN_PREVIEW_POSITION),
    ]

    for brick_type in INVENTORY_ORDER:
        for brick_index, position in iter_inventory_positions(brick_type, bounds, layout):
            brick_id = f"{brick_type.prefix}{brick_index}"
            is_dynamic = brick_id == dynamic_brick_id
            lines.append(build_brick_body_xml(brick_type, brick_id, position, static=not is_dynamic))

    lines.append("  </worldbody>")
    return "\n".join(lines) + "\n"


def build_scene_xml(bounds: WorldBounds, layout: InventoryLayout, dynamic_brick_id: Optional[str] = None) -> str:
    asset_xml = build_asset_section()
    worldbody_xml = build_worldbody_xml(bounds, layout, dynamic_brick_id)
    xml = f"""<mujoco model="lego_planning_learning_inventory">
  <compiler angle="degree" coordinate="local" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.65 0.65 0.65" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
    <rgba haze="0.97 0.98 1.0 1"/>
  </visual>
  <statistic center="{format_vec(unity_to_mujoco_position(bounds.center))}" extent="{max(bounds.length, bounds.width):.6f}"/>
{asset_xml}  <default>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001" friction="0.9 0.1 0.05" condim="3"/>
    <joint damping="1" armature="0.01"/>
  </default>
{worldbody_xml}</mujoco>
"""
    return xml


def write_scene_file(xml: str) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    xml_path = BUILD_DIR / "lego_inventory_scene.xml"
    xml_path.write_text(xml, encoding="utf-8")
    return xml_path


def summarize_layout(bounds: WorldBounds, layout: InventoryLayout) -> str:
    summary_lines = [
        "MuJoCo LEGO/PLAEX inventory scene",
        f"Bounds: length={bounds.length:.2f}, width={bounds.width:.2f}, wall_height={bounds.wall_height:.2f}",
        f"Inventory grid per group: rows={layout.rows}, columns={layout.columns}, layers={layout.layers}",
        f"Brick groups: {', '.join(brick_type.group_name for brick_type in INVENTORY_ORDER)}",
        # f"Origin preview brick: {ORIGIN_PREVIEW_BRICK_ID} at {ORIGIN_PREVIEW_POSITION}",
    ]
    return "\n".join(summary_lines)


def find_inventory_position(
    brick_id: str, bounds: WorldBounds, layout: InventoryLayout
) -> Optional[Tuple[float, float, float]]:
    """Return the MuJoCo-space position of a brick in the inventory, or None if not found."""
    for brick_type in INVENTORY_ORDER:
        for brick_index, pos_unity in iter_inventory_positions(brick_type, bounds, layout):
            if f"{brick_type.prefix}{brick_index}" == brick_id:
                return unity_to_mujoco_position(pos_unity)
    return None


def euler_to_quat(a: float, b: float, c: float) -> Tuple[float, float, float, float]:
    """Convert intrinsic XYZ Euler angles (degrees) to a MuJoCo quaternion (w, x, y, z)."""
    hr = math.radians(a) * 0.5
    hp = math.radians(b) * 0.5
    hy = math.radians(c) * 0.5
    cr, sr = math.cos(hr), math.sin(hr)
    cp, sp = math.cos(hp), math.sin(hp)
    cy, sy = math.cos(hy), math.sin(hy)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def lerp(
    p0: Tuple[float, float, float], p1: Tuple[float, float, float], t: float
) -> Tuple[float, float, float]:
    return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]), p0[2] + t * (p1[2] - p0[2]))


def slerp(
    q0: Tuple[float, float, float, float],
    q1: Tuple[float, float, float, float],
    t: float,
) -> Tuple[float, float, float, float]:
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        dot = -dot
    if dot > 0.9995:
        r = tuple(a + t * (b - a) for a, b in zip(q0, q1))
        n = math.sqrt(sum(x * x for x in r))
        return tuple(x / n for x in r)  # type: ignore[return-value]
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    s0 = math.cos(theta) - dot * math.sin(theta) / math.sin(theta_0)
    s1 = math.sin(theta) / math.sin(theta_0)
    return tuple(s0 * a + s1 * b for a, b in zip(q0, q1))  # type: ignore[return-value]


def quat_multiply(
    q1: Tuple[float, float, float, float],
    q2: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Hamilton product q1 * q2  (applies q2 in q1's local frame)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


PlanAction = Union[PickAction, OrientAction, PlaceAction]

_PICK_RE = re.compile(r'^pick\(\s*["\']([^"\']+)["\']\s*\)$', re.IGNORECASE)
_ORIENT_RE = re.compile(r'^orient\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)$', re.IGNORECASE)
_PLACE_RE = re.compile(r'^place\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)$', re.IGNORECASE)


def parse_plan(path: Path) -> List[PlanAction]:
    actions: List[PlanAction] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = _PICK_RE.match(line)
        if m:
            actions.append(PickAction(brick_id=m.group(1)))
            continue

        m = _ORIENT_RE.match(line)
        if m:
            actions.append(OrientAction(a=float(m.group(1)), b=float(m.group(2)), c=float(m.group(3))))
            continue

        m = _PLACE_RE.match(line)
        if m:
            actions.append(PlaceAction(x=float(m.group(1)), y=float(m.group(2)), z=float(m.group(3))))
            continue

        raise ValueError(f"Unrecognised instruction on line {line_no}: {raw!r}")

    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and optionally view the MuJoCo LEGO/PLAEX inventory scene.")
    parser.add_argument("--no-viewer", action="store_true", help="Only generate and validate the scene XML.")
    parser.add_argument("--dump-xml", action="store_true", help="Print the generated XML to stdout.")
    parser.add_argument("--plan", type=Path, metavar="FILE", help="Path to a .plan file to execute.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = InventoryLayout()
    bounds = compute_required_bounds(layout)

    # --- Parse plan ---
    dynamic_brick_id: Optional[str] = None
    orient_action: Optional[OrientAction] = None
    place_action: Optional[PlaceAction] = None

    if args.plan is not None:
        for action in parse_plan(args.plan):
            if isinstance(action, PickAction):
                dynamic_brick_id = action.brick_id
            elif isinstance(action, OrientAction):
                orient_action = action
            elif isinstance(action, PlaceAction):
                place_action = action

    xml = build_scene_xml(bounds, layout, dynamic_brick_id)
    xml_path = write_scene_file(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print(summarize_layout(bounds, layout))
    print(f"Generated XML: {xml_path}")
    print("Mesh support: .blend files are detected as references only; MuJoCo will use exported .obj/.stl/.msh files if present, otherwise procedural primitives are used.")

    if args.dump_xml:
        print(xml)

    if args.no_viewer:
        return

    try:
        from mujoco import viewer as mj_viewer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mujoco.viewer is unavailable in this Python environment.") from exc

    # --- Animation setup ---
    PHASE_DURATION = 4.0     # wall-clock seconds per animation phase
    LIFT_CLEARANCE = 15.0    # MuJoCo units to lift above inventory before travelling

    anim_qpos_addr: Optional[int] = None
    anim_dof_addr: Optional[int] = None
    src_pos: Optional[Tuple[float, float, float]] = None
    tgt_pos: Optional[Tuple[float, float, float]] = None
    base_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    tgt_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    if dynamic_brick_id is not None and place_action is not None:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, dynamic_brick_id)
        if body_id >= 0:
            jnt_id = model.body_jntadr[body_id]
            anim_qpos_addr = int(model.jnt_qposadr[jnt_id])
            anim_dof_addr = int(model.jnt_dofadr[jnt_id])
            # Read the quaternion MuJoCo initialised from the XML euler attribute.
            # This captures the mesh-correction rotation (e.g. euler="90 0 0") so
            # the brick's studs stay upright throughout the animation.
            q = data.qpos[anim_qpos_addr + 3:anim_qpos_addr + 7]
            base_quat = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

        src_pos = find_inventory_position(dynamic_brick_id, bounds, layout)
        tgt_pos = unity_to_mujoco_position((place_action.x, place_action.y, place_action.z))

        # Compose the user's orient on top of the base rotation so that
        # orient(0, 0, 0) means "natural upright orientation, no extra rotation".
        user_quat = euler_to_quat(orient_action.a, orient_action.b, orient_action.c) \
            if orient_action is not None else (1.0, 0.0, 0.0, 0.0)
        tgt_quat = quat_multiply(base_quat, user_quat)

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = bounds.center
        viewer.cam.distance = max(bounds.length, bounds.width) * 1.35
        viewer.cam.elevation = -28
        viewer.cam.azimuth = 90

        anim_start = time.time()

        while viewer.is_running():
            # Kinematic animation: override qpos/qvel for the picked brick each step
            if anim_qpos_addr is not None and src_pos is not None and tgt_pos is not None:
                elapsed = time.time() - anim_start
                phase = int(elapsed // PHASE_DURATION)
                t = min((elapsed % PHASE_DURATION) / PHASE_DURATION, 1.0)

                lift_src: Tuple[float, float, float] = (src_pos[0], src_pos[1], src_pos[2] + LIFT_CLEARANCE)
                lift_tgt: Tuple[float, float, float] = (tgt_pos[0], tgt_pos[1], tgt_pos[2] + LIFT_CLEARANCE)

                if phase == 0:    # lift straight up from inventory
                    pos = lerp(src_pos, lift_src, t)
                    quat = slerp(base_quat, tgt_quat, t)
                elif phase == 1:  # travel horizontally to above target
                    pos = lerp(lift_src, lift_tgt, t)
                    quat = tgt_quat
                elif phase == 2:  # lower onto target
                    pos = lerp(lift_tgt, tgt_pos, t)
                    quat = tgt_quat
                else:             # done — hold in place
                    pos = tgt_pos
                    quat = tgt_quat

                data.qpos[anim_qpos_addr:anim_qpos_addr + 3] = pos
                data.qpos[anim_qpos_addr + 3:anim_qpos_addr + 7] = quat
                data.qvel[anim_dof_addr:anim_dof_addr + 6] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
