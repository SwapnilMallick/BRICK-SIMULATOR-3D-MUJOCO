import bpy
import bmesh
import math
from mathutils import Vector


OBJ_NAME = "PLAEX_Long_Tab"


BODY_X = 5
BODY_Y = 2
BODY_Z = 2


TAB_BOTTOM_WIDTH = 1.0
TAB_TOP_WIDTH = 0.5
TAB_DEPTH = 0.25
TAB_FULL_HEIGHT = 2.0


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
        bsdf.inputs["Base Color"].default_value = (1.0, 0.9, 0.0, 1.0)
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
        

def build_outer_body():
    obj, mesh = create_mesh_object(OBJ_NAME)
    bm = bmesh.new()

    #add_box(bm, BODY_X, BODY_Y, BODY_Z, (0.0, 0.0, 0.0))
    add_trapezoid_prism(bm, center_xy=(-3.4, 0.0), axis="x", outward_sign=-1)
    #add_trapezoid_prism(bm, center_xy=(-1.25, 3.4), axis="y", outward_sign=1)
    #add_trapezoid_prism(bm, center_xy=(1.25, 3.4), axis="y", outward_sign=1)
    #add_trapezoid_prism(bm, center_xy=(-1.25, -3.4), axis="y", outward_sign=-1)
    #add_trapezoid_prism(bm, center_xy=(1.25, -3.4), axis="y", outward_sign=-1)

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


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

    body = build_outer_body()
    assign_material(body, material)

    #subtract_side_grooves(body)

    #stud_inner_radius = STUD_RADIUS - STUD_WALL
    #ubtract_internal_cutters(body, stud_inner_radius)

    #studs = build_top_studs(material)

    final_obj = join_objects([body], OBJ_NAME, material)
    cleanup_final_mesh(final_obj)
    assign_material(final_obj, material)
    print_mesh_health(final_obj)


if __name__ == "__main__":
    main()