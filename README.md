# LEGO/PLAEX MuJoCo Simulator

This workspace contains a first MuJoCo recreation of the Unity simulator's world bounds and inventory setup.

## What It Mirrors

- The bounded world with one floor and four surrounding walls.
- The same seven inventory groups as the Unity simulator:
  - `Inventory_NormalBricks`
  - `Inventory_LegoBricks`
  - `Inventory_OrangeLegoBricks`
  - `Inventory_GreenPlaexLongBricks`
  - `Inventory_OrangePlaexLongBricks`
  - `Inventory_YellowPlaexLongBricks`
  - `Inventory_YellowPlaexSideBricks`
- The same inventory grid dimensions and spacing:
  - `rows=5`
  - `columns=5`
  - `layers=10`
  - `gap_x=gap_y=gap_z=0.5`
  - `inventory_separation_gap=5`
  - `inventory_corner_inset=1`

## Running

```bash
python3 simulator.py
```

To only generate and validate the MuJoCo XML without opening the interactive viewer:

```bash
python3 simulator.py --no-viewer
```

The generated XML is written to:

`build/lego_inventory_scene.xml`

## About The Blender Files

The source Blender assets live in:

`BlenderFiles/`

MuJoCo cannot consume `.blend` files directly. The simulator checks for exported mesh files with the same base names and these extensions:

- `.obj`
- `.stl`
- `.msh`

If those exported files are present, the simulator will use them automatically. Otherwise it falls back to procedural MuJoCo primitives that approximate the Unity inventory bricks.

## Current Brick Approximations

- Normal bricks: box, `5 x 2 x 2`
- Procedural LEGO bricks: box with studs, `4 x 0.96 x 1.6`
- Orange LEGO bricks: box with studs, `4 x 0.96 x 1.6`
- PLAEX long bricks: box, `5 x 2 x 2`
- PLAEX side bricks: box, `2 x 2 x 2`

## Next Logical Steps

- Export the Blender assets to `.obj` or `.stl` and wire in per-brick mesh transforms if needed.
- Add plan loading and placement execution.
- Add LEGO and PLAEX snap logic.
