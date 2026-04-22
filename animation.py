from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mujoco


PHASE_DURATION = 1.2      # wall-clock seconds per animation phase
PHASES_PER_BRICK = 3      # arc-to-above-target → lower-to-release → physics settle
LIFT_CLEARANCE = 15.0     # MuJoCo units above the higher of src/tgt during arc
RELEASE_CLEARANCE = 0.5   # MuJoCo units above tgt_pos where kinematic drive ends.
                           # Zero means phase 1 delivers the brick exactly to tgt_pos,
                           # so phase 2 physics begins from rest at the plan position.
                           # This eliminates the tab/groove lateral contact force that
                           # arises when a falling brick hits a frozen neighbour, and
                           # avoids the airborne-brick problem caused by the brick
                           # not having enough simulation time to fall 1 unit before
                           # its settled pose is captured.


@dataclass
class BrickAnimState:
    qpos_addr: int
    dof_addr: int
    src_pos: Tuple[float, float, float]
    tgt_pos: Tuple[float, float, float]
    base_quat: Tuple[float, float, float, float]
    tgt_quat: Tuple[float, float, float, float]
    # Settled pose captured once at the end of phase 2; None until then.
    frozen_qpos: Optional[Tuple[float, ...]] = None


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


def smoothstep(t: float) -> float:
    """Smooth ease-in / ease-out: zero first-derivative at t=0 and t=1."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def bezier3(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    """Quadratic Bezier: P(t) = (1-t)²·P0 + 2(1-t)t·P1 + t²·P2."""
    mt = 1.0 - t
    return (
        mt * mt * p0[0] + 2.0 * mt * t * p1[0] + t * t * p2[0],
        mt * mt * p0[1] + 2.0 * mt * t * p1[1] + t * t * p2[1],
        mt * mt * p0[2] + 2.0 * mt * t * p1[2] + t * t * p2[2],
    )


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


def build_anim_bricks(
    triplets,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    find_inventory_position,
    bounds,
    layout,
) -> List[BrickAnimState]:
    """Build the list of per-brick animation states from a parsed plan's triplets.

    Args:
        triplets: List of (PickAction, Optional[OrientAction], Optional[PlaceAction]).
        model: The MuJoCo model.
        data: The MuJoCo data (used to read the initial quaternion set by the XML).
        find_inventory_position: Callable(brick_id, bounds, layout) -> Optional[pos].
        bounds: WorldBounds instance.
        layout: InventoryLayout instance.
    """
    anim_bricks: List[BrickAnimState] = []

    for pick_action, orient_action, place_action in triplets:
        if place_action is None:
            continue
        brick_id = pick_action.brick_id
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, brick_id)
        if body_id < 0:
            continue
        jnt_id = model.body_jntadr[body_id]
        qpos_addr = int(model.jnt_qposadr[jnt_id])
        dof_addr = int(model.jnt_dofadr[jnt_id])
        # Read the quaternion MuJoCo initialised from the XML euler attribute.
        # This captures the mesh-correction rotation (e.g. euler="90 0 0") so
        # the brick's studs stay upright throughout the animation.
        q = data.qpos[qpos_addr + 3:qpos_addr + 7]
        base_quat: Tuple[float, float, float, float] = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        src_pos = find_inventory_position(brick_id, bounds, layout)
        if src_pos is None:
            continue
        tgt_pos = (place_action.x, place_action.y, place_action.z)
        # Compose the user's orient on top of the base rotation so that
        # orient(0, 0, 0) means "natural upright orientation, no extra rotation".
        user_quat = euler_to_quat(orient_action.a, orient_action.b, orient_action.c) \
            if orient_action is not None else (1.0, 0.0, 0.0, 0.0)
        tgt_quat = quat_multiply(base_quat, user_quat)
        anim_bricks.append(BrickAnimState(qpos_addr, dof_addr, src_pos, tgt_pos, base_quat, tgt_quat))

    return anim_bricks


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bounds,
    anim_bricks: List[BrickAnimState],
) -> None:
    """Launch the MuJoCo passive viewer and run the brick animation loop."""
    try:
        from mujoco import viewer as mj_viewer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mujoco.viewer is unavailable in this Python environment.") from exc

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = bounds.center
        viewer.cam.distance = max(bounds.length, bounds.width) * 1.35
        viewer.cam.elevation = -28
        viewer.cam.azimuth = 90

        anim_start = time.time()

        while viewer.is_running():
            if anim_bricks:
                elapsed = time.time() - anim_start
                # Which brick is currently animating, and which phase within it.
                global_phase = int(elapsed // PHASE_DURATION)
                brick_idx   = global_phase // PHASES_PER_BRICK
                local_phase = global_phase % PHASES_PER_BRICK
                t = min((elapsed % PHASE_DURATION) / PHASE_DURATION, 1.0)

                # Kinematically lock every brick that has finished phase 2.
                # X/Y and orientation are taken from the plan (they were pinned
                # throughout phase 2, so no lateral drift can have occurred).
                # Z is taken from the physics-settled qpos so the contact-resolved
                # resting height is preserved without a visual jump.
                for past_bk in anim_bricks[:brick_idx]:
                    if past_bk.frozen_qpos is None:
                        past_bk.frozen_qpos = tuple(data.qpos[past_bk.qpos_addr:past_bk.qpos_addr + 7])
                    data.qpos[past_bk.qpos_addr:past_bk.qpos_addr + 7] = past_bk.frozen_qpos
                    data.qvel[past_bk.dof_addr:past_bk.dof_addr + 6] = 0.0

                # Animate the current brick: phases 0–1 are kinematic, phase 2 is pure physics.
                if brick_idx < len(anim_bricks):
                    bk = anim_bricks[brick_idx]

                    # Apex of the arc: above whichever end-point is higher, so
                    # the brick never dips below either during transit.
                    arc_peak_z = max(bk.src_pos[2], bk.tgt_pos[2]) + LIFT_CLEARANCE
                    arc_ctrl: Tuple[float, float, float] = (
                        (bk.src_pos[0] + bk.tgt_pos[0]) * 0.5,
                        (bk.src_pos[1] + bk.tgt_pos[1]) * 0.5,
                        arc_peak_z,
                    )
                    # Where the arc ends: directly above the target at the same
                    # lift height, so phase 1 only has to descend vertically.
                    lift_tgt: Tuple[float, float, float] = (
                        bk.tgt_pos[0], bk.tgt_pos[1], arc_peak_z)
                    # Intermediate waypoint: RELEASE_CLEARANCE above the target.
                    # Phase 1 ends here; phase 2 covers the final descent.
                    release_pos: Tuple[float, float, float] = (
                        bk.tgt_pos[0], bk.tgt_pos[1], bk.tgt_pos[2] + RELEASE_CLEARANCE)

                    if local_phase == 0:
                        # Quadratic Bezier arc: src → ctrl (apex) → lift_tgt.
                        # The brick lifts, sweeps across, and arrives directly
                        # above the target in one smooth curved motion.
                        # Orientation is eased in during this phase so it
                        # arrives aligned before the final descent.
                        t_s = smoothstep(t)
                        pos  = bezier3(bk.src_pos, arc_ctrl, lift_tgt, t_s)
                        quat = slerp(bk.base_quat, bk.tgt_quat, t_s)
                        data.qpos[bk.qpos_addr:bk.qpos_addr + 3]     = pos
                        data.qpos[bk.qpos_addr + 3:bk.qpos_addr + 7] = quat
                        data.qvel[bk.dof_addr:bk.dof_addr + 6]       = 0.0
                    elif local_phase == 1:
                        # Ease down from lift height to the release waypoint.
                        t_s = smoothstep(t)
                        pos = lerp(lift_tgt, release_pos, t_s)
                        data.qpos[bk.qpos_addr:bk.qpos_addr + 3]     = pos
                        data.qpos[bk.qpos_addr + 3:bk.qpos_addr + 7] = bk.tgt_quat
                        data.qvel[bk.dof_addr:bk.dof_addr + 6]       = 0.0
                    elif local_phase == 2:
                        # Pure physics settle.  Phase 1 delivered the brick to
                        # tgt_pos at rest (RELEASE_CLEARANCE = 0), so contact
                        # forces are zero at the start of this phase and no
                        # lateral impulse is generated.  Unstable bricks will
                        # topple naturally; stud-tube and tab-groove contacts
                        # engage under gravity without kinematic interference.
                        pass

                # Freeze every brick that hasn't started animating yet.
                # All plan bricks have a freejoint, so without this they would
                # fall off the shelf under gravity before their turn arrives.
                for future_bk in anim_bricks[brick_idx + 1:]:
                    data.qpos[future_bk.qpos_addr:future_bk.qpos_addr + 3]     = future_bk.src_pos
                    data.qpos[future_bk.qpos_addr + 3:future_bk.qpos_addr + 7] = future_bk.base_quat
                    data.qvel[future_bk.dof_addr:future_bk.dof_addr + 6]        = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()
