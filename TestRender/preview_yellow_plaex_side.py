from __future__ import annotations

import argparse
from pathlib import Path

import mujoco


WORKSPACE_ROOT = Path(__file__).resolve().parent
BUILD_DIR = WORKSPACE_ROOT / "build"
BLENDER_DIR = WORKSPACE_ROOT / "BlenderFiles"
MESH_PATH = BLENDER_DIR / "yellowPLAEXSide.obj"

BRICK_POSITION = (0.0, 1.0, 5.0)
BRICK_RGBA = (0.98, 0.86, 0.12, 1.0)
FALLBACK_HALF_EXTENTS = (1.0, 1.0, 1.0)


def format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def build_scene_xml() -> str:
    asset_xml = "  <asset/>\n"
    geom_xml = (
        f'      <geom name="yellow_plaex_side_geom" type="box" size="{format_vec(FALLBACK_HALF_EXTENTS)}" '
        f'rgba="{format_vec(BRICK_RGBA)}"/>\n'
    )

    if MESH_PATH.exists():
        asset_xml = (
            "  <asset>\n"
            f'    <mesh name="yellow_plaex_side_mesh" file="{MESH_PATH.relative_to(WORKSPACE_ROOT).as_posix()}"/>\n'
            "  </asset>\n"
        )
        geom_xml = (
            f'      <geom name="yellow_plaex_side_geom" type="mesh" mesh="yellow_plaex_side_mesh" '
            f'rgba="{format_vec(BRICK_RGBA)}"/>\n'
        )

    return f"""<mujoco model="yellow_plaex_side_preview">
  <compiler angle="degree" coordinate="local" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 0"/>
  <visual>
    <headlight ambient="0.7 0.7 0.7" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <rgba haze="0.97 0.98 1.0 1"/>
  </visual>
  <statistic center="{format_vec(BRICK_POSITION)}" extent="15"/>
{asset_xml}  <worldbody>
    <light name="key" pos="0 -8 10" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="preview" pos="0 -12 6" xyaxes="1 0 0 0 0.6 0.8"/>
    <geom name="floor" type="box" pos="0 0 -0.1" size="8 8 0.1" rgba="0.85 0.87 0.91 1"/>
    <body name="PreviewYellowPlaexSideBrick" pos="{format_vec(BRICK_POSITION)}">
{geom_xml}    </body>
  </worldbody>
</mujoco>
"""


def write_scene(xml: str) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BUILD_DIR / "yellow_plaex_side_preview.xml"
    output_path.write_text(xml, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a standalone preview scene for the yellow PLAEX side brick.")
    parser.add_argument("--no-viewer", action="store_true", help="Only generate and validate the scene XML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml = build_scene_xml()
    output_path = write_scene(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print(f"Generated preview scene: {output_path}")
    print(f"Brick position: {BRICK_POSITION}")
    print(f"Using mesh: {MESH_PATH.exists()}")

    if args.no_viewer:
        return

    try:
        from mujoco import viewer as mj_viewer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mujoco.viewer is unavailable in this Python environment.") from exc

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = BRICK_POSITION
        viewer.cam.distance = 12
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
