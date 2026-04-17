import bpy
import bmesh
import math
from mathutils import Vector


# Procedural Yellow PLAEX Long brick rebuild.
# Feature set from the original yellow script:
# - Base 5x2x1 body
# - One outward tab on -X
# - Two outward tabs on +Y
# - Two outward tabs on -Y
# - One trapezoidal groove on +X
# - Two hollow top studs
# - Through-holes under each stud
# - Two underside recesses


OBJ_NAME = "PLAEXLong_5x2x1"

BODY_X = 5.0
BODY_Y = 2.0
BODY_Z = 1.0

TAB_BOTTOM_WIDTH = 1.0
TAB_TOP_WIDTH = 0.5
TAB_DEPTH = 0.25
TAB_FULL_HEIGHT = 1.0

GROOVE_BOTTOM_WIDTH = 1.0
GROOVE_TOP_WIDTH = 0.5
GROOVE_DEPTH = 0.25
GROOVE_HEIGHT = 1.0
GROOVE_CLEARANCE = 0.03
GROOVE_EPSILON_OPEN = 0.002

STUD_RADIUS = 0.65
STUD_HEIGHT = 0.30
STUD_WALL = 0.12
STUD_SEGMENTS = 48
STUD_POSITIONS = [(-1.25, 0.0), (1.25, 0.0)]

BOTTOM_RECESS_RADIUS = STUD_RADIUS + 0.03
BOTTOM_RECESS_DEPTH = 0.34
BOTTOM_RECESS_SEGMENTS = 48


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def activate_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def ensure_material():
    mat = bpy.data.materials.get("YellowMaterial")
    if mat is None:
        mat = bpy.data.materials.new(name="YellowMaterial")
        mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = 1.0

    if hasattr(mat, "blend_method"):
        mat.blend_method = "OPAQUE"

    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "OPAQUE"

    return mat


def assign_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def create_mesh_object(name):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj, mesh


def apply_boolean(target_obj, tool_obj, operation, name):
    activate_only(target_obj)
    mod = target_obj.modifiers.new(name=name, type="BOOLEAN")
    mod.operation = operation
    mod.solver = "EXACT"
    mod.object = tool_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(tool_obj, do_unlink=True)


def add_box(bm, size_x, size_y, size_z, center):
    result = bmesh.ops.create_cube(bm, size=1.0)
    verts = result["verts"]
    cx, cy, cz = center

    for v in verts:
        v.co.x = cx + (v.co.x * size_x)
        v.co.y = cy + (v.co.y * size_y)
        v.co.z = cz + (v.co.z * size_z)

    return verts


def add_trapezoid_prism(bm, center_xy, axis, outward_sign):
    cx, cy = center_xy
    hz = TAB_FULL_HEIGHT * 0.5
    bw = TAB_BOTTOM_WIDTH * 0.5
    tw = TAB_TOP_WIDTH * 0.5

    if axis == "x":
        inner_x = outward_sign * (BODY_X * 0.5)
        outer_x = inner_x + (outward_sign * TAB_DEPTH)
        bottom = [
            (inner_x, cy - tw, -hz),
            (outer_x, cy - bw, -hz),
            (outer_x, cy + bw, -hz),
            (inner_x, cy + tw, -hz),
        ]
        top = [
            (inner_x, cy - tw, hz),
            (outer_x, cy - bw, hz),
            (outer_x, cy + bw, hz),
            (inner_x, cy + tw, hz),
        ]
    else:
        inner_y = outward_sign * (BODY_Y * 0.5)
        outer_y = inner_y + (outward_sign * TAB_DEPTH)
        bottom = [
            (cx - tw, inner_y, -hz),
            (cx - bw, outer_y, -hz),
            (cx + bw, outer_y, -hz),
            (cx + tw, inner_y, -hz),
        ]
        top = [
            (cx - tw, inner_y, hz),
            (cx - bw, outer_y, hz),
            (cx + bw, outer_y, hz),
            (cx + tw, inner_y, hz),
        ]

    verts = [bm.verts.new(Vector(co)) for co in bottom + top]
    bm.verts.ensure_lookup_table()

    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    for face_indices in faces:
        try:
            bm.faces.new([verts[i] for i in face_indices])
        except ValueError:
            pass


def circle_points(radius, segments, z, center_xy):
    cx, cy = center_xy
    points = []
    for i in range(segments):
        angle = (2.0 * math.pi * i) / segments
        points.append(Vector((cx + (radius * math.cos(angle)), cy + (radius * math.sin(angle)), z)))
    return points


def add_ring_section(bm, outer_radius, inner_radius, z0, z1, center_xy, segments, cap_top=True, cap_bottom=True):
    bottom_outer = [bm.verts.new(co) for co in circle_points(outer_radius, segments, z0, center_xy)]
    top_outer = [bm.verts.new(co) for co in circle_points(outer_radius, segments, z1, center_xy)]
    bottom_inner = [bm.verts.new(co) for co in circle_points(inner_radius, segments, z0, center_xy)]
    top_inner = [bm.verts.new(co) for co in circle_points(inner_radius, segments, z1, center_xy)]
    bm.verts.ensure_lookup_table()

    for i in range(segments):
        j = (i + 1) % segments

        for quad in (
            [bottom_outer[i], bottom_outer[j], top_outer[j], top_outer[i]],
            [bottom_inner[j], bottom_inner[i], top_inner[i], top_inner[j]],
        ):
            try:
                bm.faces.new(quad)
            except ValueError:
                pass

        if cap_top:
            try:
                bm.faces.new([top_outer[i], top_outer[j], top_inner[j], top_inner[i]])
            except ValueError:
                pass

        if cap_bottom:
            try:
                bm.faces.new([bottom_outer[j], bottom_outer[i], bottom_inner[i], bottom_inner[j]])
            except ValueError:
                pass


def build_outer_body():
    obj, mesh = create_mesh_object(OBJ_NAME)
    bm = bmesh.new()

    add_box(bm, BODY_X, BODY_Y, BODY_Z, (0.0, 0.0, 0.0))
    add_trapezoid_prism(bm, center_xy=(-3.4, 0.0), axis="x", outward_sign=-1)
    add_trapezoid_prism(bm, center_xy=(-1.25, 3.4), axis="y", outward_sign=1)
    add_trapezoid_prism(bm, center_xy=(1.25, 3.4), axis="y", outward_sign=1)
    add_trapezoid_prism(bm, center_xy=(-1.25, -3.4), axis="y", outward_sign=-1)
    add_trapezoid_prism(bm, center_xy=(1.25, -3.4), axis="y", outward_sign=-1)

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def create_trapezoid_groove_cutter(name, center_xy, target_x, rotation_z_degrees):
    cut_bottom_width = GROOVE_BOTTOM_WIDTH + (2.0 * GROOVE_CLEARANCE)
    cut_top_width = GROOVE_TOP_WIDTH + (2.0 * GROOVE_CLEARANCE)
    cut_depth = GROOVE_DEPTH + GROOVE_CLEARANCE
    cut_height = GROOVE_HEIGHT + (2.0 * GROOVE_CLEARANCE)

    half_h = cut_height * 0.5
    half_d = cut_depth * 0.5
    bw = cut_bottom_width * 0.5
    tw = cut_top_width * 0.5
    cx, cy = center_xy

    verts = [
        (cx - bw, cy - half_d, -half_h),
        (cx + bw, cy - half_d, -half_h),
        (cx + tw, cy + half_d, -half_h),
        (cx - tw, cy + half_d, -half_h),
        (cx - bw, cy - half_d, half_h),
        (cx + bw, cy - half_d, half_h),
        (cx + tw, cy + half_d, half_h),
        (cx - tw, cy + half_d, half_h),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    cutter = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(cutter)

    activate_only(cutter)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    cutter.rotation_euler[2] = math.radians(rotation_z_degrees)
    bpy.context.view_layer.update()

    small_face_ids = [2, 3, 6, 7]
    small_face_world_x = sum(
        (cutter.matrix_world @ Vector(cutter.data.vertices[i].co)).x
        for i in small_face_ids
    ) / 4.0

    if target_x > 0:
        cutter.location.x += (target_x + GROOVE_EPSILON_OPEN - small_face_world_x)
    else:
        cutter.location.x += (target_x - GROOVE_EPSILON_OPEN - small_face_world_x)

    activate_only(cutter)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return cutter


def subtract_side_grooves(body_obj):
    pos_cutter = create_trapezoid_groove_cutter(
        "TrapeziumCutter_PosX",
        center_xy=(3.4, 0.0),
        target_x=(BODY_X * 0.5),
        rotation_z_degrees=-90,
    )
    apply_boolean(body_obj, pos_cutter, "DIFFERENCE", "PosX_TrapeziumGroove")


def subtract_internal_cutters(body_obj, stud_inner_radius):
    top_face_z = BODY_Z * 0.5

    hole_top_z = top_face_z + STUD_HEIGHT + 0.02
    hole_bottom_z = top_face_z - 2.0
    through_depth = hole_top_z - hole_bottom_z
    through_center_z = (hole_top_z + hole_bottom_z) * 0.5

    recess_bottom_z = (-BODY_Z * 0.5) - 0.02
    recess_top_z = recess_bottom_z + BOTTOM_RECESS_DEPTH + 0.02
    recess_depth = recess_top_z - recess_bottom_z
    recess_center_z = (recess_top_z + recess_bottom_z) * 0.5

    for i, (x, y) in enumerate(STUD_POSITIONS, start=1):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=STUD_SEGMENTS,
            radius=stud_inner_radius,
            depth=through_depth,
            location=(x, y, through_center_z),
        )
        through_cutter = bpy.context.active_object
        through_cutter.name = f"ThroughHoleCutter_{i}"
        apply_boolean(body_obj, through_cutter, "DIFFERENCE", f"ThroughHole_{i}")

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=BOTTOM_RECESS_SEGMENTS,
            radius=BOTTOM_RECESS_RADIUS,
            depth=recess_depth,
            location=(x, y, recess_center_z),
        )
        recess_cutter = bpy.context.active_object
        recess_cutter.name = f"BottomRecessCutter_{i}"
        apply_boolean(body_obj, recess_cutter, "DIFFERENCE", f"BottomRecess_{i}")


def cleanup_final_mesh(obj):
    activate_only(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    try:
        bpy.ops.mesh.merge_by_distance()
    except Exception:
        bpy.ops.mesh.remove_doubles()

    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def print_mesh_health(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    boundary_edges = [e for e in bm.edges if e.is_boundary]
    non_manifold_edges = [e for e in bm.edges if not e.is_manifold]
    print("Boundary edges:", len(boundary_edges))
    print("Non-manifold edges:", len(non_manifold_edges))
    bm.free()


def join_objects(objects, name, material):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = name
    assign_material(joined, material)
    return joined


def build_top_studs(material):
    stud_obj, stud_mesh = create_mesh_object("TopStuds")
    bm_studs = bmesh.new()

    top_z0 = BODY_Z * 0.5
    top_z1 = top_z0 + STUD_HEIGHT
    stud_inner_radius = STUD_RADIUS - STUD_WALL

    for center_xy in STUD_POSITIONS:
        add_ring_section(
            bm_studs,
            outer_radius=STUD_RADIUS,
            inner_radius=stud_inner_radius,
            z0=top_z0,
            z1=top_z1,
            center_xy=center_xy,
            segments=STUD_SEGMENTS,
            cap_top=True,
            cap_bottom=True,
        )

    bmesh.ops.remove_doubles(bm_studs, verts=bm_studs.verts, dist=0.0001)
    bmesh.ops.recalc_face_normals(bm_studs, faces=bm_studs.faces)
    bm_studs.to_mesh(stud_mesh)
    bm_studs.free()
    stud_mesh.update()
    assign_material(stud_obj, material)
    return stud_obj


def main():
    clear_scene()
    material = ensure_material()

    body = build_outer_body()
    assign_material(body, material)

    subtract_side_grooves(body)

    stud_inner_radius = STUD_RADIUS - STUD_WALL
    subtract_internal_cutters(body, stud_inner_radius)

    studs = build_top_studs(material)

    final_obj = join_objects([body, studs], OBJ_NAME, material)
    cleanup_final_mesh(final_obj)
    assign_material(final_obj, material)
    print_mesh_health(final_obj)


if __name__ == "__main__":
    main()
