# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A 3D physics-based brick manipulation simulator using MuJoCo. It simulates PLAEX brick assembly — reading `.plan` files that describe pick/orient/place action sequences, generating a MuJoCo XML scene procedurally, animating bricks from an inventory grid to target positions, and letting physics settle them into place.

## Running the Simulator

```bash
# Launch with interactive MuJoCo viewer (default)
python3 simulator.py

# Execute a plan file
python3 simulator.py --plan Plans/placeA2Row.plan

# Generate XML only, no viewer
python3 simulator.py --no-viewer

# Dump generated XML to stdout
python3 simulator.py --dump-xml
```

No install step — the only dependency is the `mujoco` Python package.

## Architecture

### Core Files

**[simulator.py](simulator.py)** — Scene builder and main orchestrator. Defines all brick types as `BrickType` dataclasses (geometry, collision, studs, tabs, grooves), generates MuJoCo XML procedurally, and runs `main()` which ties everything together.

**[animation.py](animation.py)** — Kinematic animation engine. Drives bricks through three phases: (0) quadratic Bezier arc lift+sweep with SLERP rotation, (1) linear descent to release height, (2) pure physics settle. Called by `run_viewer()` which owns the MuJoCo passive viewer loop.

**[plan_parser.py](plan_parser.py)** — Regex-based parser for the `.plan` DSL. Converts `pick()`, `orient()`, `place()`, and `moveBrick()` statements into typed action objects.

### Execution Flow

```
main()
 ├─ Parse CLI args
 ├─ Compute world bounds from inventory layout
 ├─ parse_plan() → list of (PickAction, OrientAction, PlaceAction) triplets
 ├─ Identify which brick IDs are dynamic (need freejoint for physics)
 ├─ build_scene_xml() → full MuJoCo XML string
 ├─ Write to build/lego_inventory_scene.xml
 ├─ mujoco.MjModel.from_xml_string() — validates XML
 ├─ Build BrickAnimState per moved brick
 └─ run_viewer() — animation + physics loop
```

### Brick Model System

Four active brick types (`INVENTORY_ORDER` in simulator.py): **A2** (5×2×2 yellow long), **C2** (2×2×2 yellow side), **A1** (5×2×1 red thin long), **C1** (2×2×1 red thin side).

Each `BrickType` specifies:
- Body box dimensions (trimmed for grooves)
- `stud_geom_xz` / `stud_radius` / `stud_height` — cylindrical protrusions on top for vertical stacking
- `tabs` — rectangular boxes approximating trapezoidal edge connectors (side-snap)
- `groove` — recessed channel on one face that receives tabs
- `groove_guards` — geometry preventing misaligned tab insertion

**Tube receiver geometry** is not a `BrickType` field — it is generated automatically inside `build_brick_body_xml()` (simulator.py:449–483) for every stud position. Per stud, four box pillars are placed on the **underside** of the brick, arranged around the stud centre to form a tube cavity that captures the stud of the brick below during vertical stacking. All four active brick types have `stud_geom_xz` defined, so all of them get tube geometry: A2 and A1 get two tubes (one per stud), C2 and C1 get one tube each.

Studs and tube pillars both use `contype="2" conaffinity="2"` so they only interact with each other — not with body boxes — letting a stud pass through the solid body box of the brick above and engage the tube instead of resting on top.

Collision geometry uses MuJoCo `contype`/`conaffinity` for selective interactions between connectors.

Mesh support: if a `.obj`/`.stl`/`.msh` exists in `BlenderFiles/` matching a brick's stem name, it's used for visuals; otherwise procedural primitives are used.

### Coordinate Conventions

- **MuJoCo world**: X = length, Y = depth, Z = up
- **OBJ/Blender** meshes are imported with `euler="90 0 0"` (Blender Y-up → MuJoCo Z-up)
- **Euler angles** in `.plan` files and `OrientAction`: intrinsic XYZ order (degrees), converted to MuJoCo quaternion (w, x, y, z) via `euler_to_quat()` in animation.py

### Inventory Layout

Bricks are arranged in a grid (`InventoryLayout`): 5 rows × 5 columns × 10 layers = 250 bricks per type. Each type occupies a separate region separated by 5-unit gaps. Brick IDs follow the pattern `Inventory{Type}_{number}` (e.g. `InventoryA2_250`).

## Plan File Format

```
pick("InventoryA2_250")
orient(0.0, 0.0, 0.0)    # Euler XYZ degrees
place(0.0, 0.0, 1.0)     # MuJoCo world coordinates

# Shorthand form:
moveBrick("InventoryA2_249", 0.0, 0.0, 90.0, 5.0, 0.0, 1.0)
```

Example plans are in [Plans/](Plans/).

## Physics Configuration

```xml
<option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>
<geom solref="0.004 1" solimp="0.95 0.99 0.001" friction="0.9 0.1 0.05"/>
```

200 Hz simulation. Instability warnings (NaN/Inf at certain DOFs) are logged to `MUJOCO_LOG.TXT` — these are known and tracked.

## Known Limitations

- No snap validation: the simulator animates bricks to positions but does not enforce or validate PLAEX/LEGO connector engagement logic.
- Blender mesh export is manual — `.obj` files in `BlenderFiles/` must be exported by hand from the `.blend` files.
- Physics instability at some DOF configurations (see `MUJOCO_LOG.TXT` and `clarifications.txt`).
