import bpy
import bmesh
import math
from mathutils import Vector


# Fully procedural Green PLAEX Long brick rebuild.
# This version removes the remaining stud/recess booleans and builds the
# whole asset directly with bmesh primitives and stitched face loops.


OBJ_NAME = "PLAEXLong_5x2x2"

BODY_X = 5.0
BODY_Y = 2.0
BODY_Z = 2.0

TAB_BOTTOM_WIDTH = 1.0
TAB_TOP_WIDTH = 0.5
TAB_DEPTH = 0.25
TAB_FULL_HEIGHT = 2.0

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
        bsdf.inputs["Base Color"].default_value = (0.1, 0.9, 0.0, 1.0)
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

        try:
            bm.faces.new([bottom_outer[i], bottom_outer[j], top_outer[j], top_outer[i]])
        except ValueError:
            pass

        try:
            bm.faces.new([bottom_inner[j], bottom_inner[i], top_inner[i], top_inner[j]])
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


def add_cylinder_wall(bm, radius, z0, z1, center_xy, segments):
    bottom_ring = [bm.verts.new(co) for co in circle_points(radius, segments, z0, center_xy)]
    top_ring = [bm.verts.new(co) for co in circle_points(radius, segments, z1, center_xy)]
    bm.verts.ensure_lookup_table()

    for i in range(segments):
        j = (i + 1) % segments

        try:
            bm.faces.new([bottom_ring[i], bottom_ring[j], top_ring[j], top_ring[i]])
        except ValueError:
            pass


def build_outer_shell():
    obj, mesh = create_mesh_object(OBJ_NAME)
    bm = bmesh.new()

    add_box(bm, BODY_X, BODY_Y, BODY_Z, (0.0, 0.0, 0.0))
    add_trapezoid_prism(bm, center_xy=(-3.4, 0.0), axis="x", outward_sign=-1)
    add_trapezoid_prism(bm, center_xy=(3.4, 0.0), axis="x", outward_sign=1)
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


def delete_faces_in_cylindrical_region(obj, center_xy, radius, z_min, z_max, normal_axis=None, normal_sign=None):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    to_delete = []
    center = Vector((center_xy[0], center_xy[1], 0.0))

    for face in bm.faces:
        face_center = face.calc_center_median()
        radial = Vector((face_center.x, face_center.y, 0.0)) - center
        in_radius = radial.length <= radius
        in_height = z_min <= face_center.z <= z_max

        if not (in_radius and in_height):
            continue

        if normal_axis is not None:
            axis_value = getattr(face.normal, normal_axis)
            if normal_sign > 0 and axis_value < 0.5:
                continue
            if normal_sign < 0 and axis_value > -0.5:
                continue

        to_delete.append(face)

    if to_delete:
        bmesh.ops.delete(bm, geom=to_delete, context="FACES")

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


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


def subtract_internal_cutters(body_obj, stud_inner_radius):
    hole_top_z = (BODY_Z * 0.5) + STUD_HEIGHT + 0.02
    hole_bottom_z = (-BODY_Z * 0.5) - 0.02
    through_depth = hole_top_z - hole_bottom_z
    through_center_z = (hole_top_z + hole_bottom_z) * 0.5

    recess_bottom_z = (-BODY_Z * 0.5) - 0.02
    recess_top_z = recess_bottom_z + BOTTOM_RECESS_DEPTH + 0.02
    recess_depth = recess_top_z - recess_bottom_z
    recess_center_z = (recess_top_z + recess_bottom_z) * 0.5

    for i, center_xy in enumerate(STUD_POSITIONS, start=1):
        x, y = center_xy

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


def main():
    clear_scene()
    material = ensure_material()

    body = build_outer_shell()
    assign_material(body, material)

    stud_inner_radius = STUD_RADIUS - STUD_WALL
    subtract_internal_cutters(body, stud_inner_radius)

    stud_obj, stud_mesh = create_mesh_object("TopStuds")
    bm_studs = bmesh.new()
    top_z0 = BODY_Z * 0.5
    top_z1 = top_z0 + STUD_HEIGHT
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

    final_obj = join_objects([body, stud_obj], OBJ_NAME, material)

    cleanup_final_mesh(final_obj)
    assign_material(final_obj, material)
    print_mesh_health(final_obj)


if __name__ == "__main__":
    main()
