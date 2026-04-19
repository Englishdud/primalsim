"""Panda3D procedural geometry helpers for the Primal Survival Simulation.

No external model files are required.  All shapes are constructed from raw
GeomVertexData / GeomTriangles so the project remains self-contained.
"""

import math
import numpy as np

from panda3d.core import (
    Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
    GeomVertexWriter, NodePath, LVecBase4f,
)


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

def make_box_np(sx: float, sy: float, sz: float, color: tuple) -> NodePath:
    """Create a solid-coloured box NodePath with proper face normals.

    Args:
        sx, sy, sz: Half-extents along X, Y, Z.
        color:      (r, g, b, a) tuple.

    Returns:
        A new NodePath ready to reparentTo() a scene graph node.
    """
    fmt   = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData('box', fmt, Geom.UHStatic)
    vdata.setNumRows(24)  # 4 verts × 6 faces

    vertex    = GeomVertexWriter(vdata, 'vertex')
    normal_w  = GeomVertexWriter(vdata, 'normal')
    color_w   = GeomVertexWriter(vdata, 'color')

    r, g, b, a = (float(c) for c in color)

    # Face definitions: list of (4 vertices, face normal)
    # Vertices listed counter-clockwise from outside the face.
    faces = [
        # +X face
        [( sx, -sy, -sz), ( sx,  sy, -sz), ( sx,  sy,  sz), ( sx, -sy,  sz)], (1, 0, 0),
        # -X face
        [(-sx,  sy, -sz), (-sx, -sy, -sz), (-sx, -sy,  sz), (-sx,  sy,  sz)], (-1, 0, 0),
        # +Y face
        [( sx,  sy, -sz), (-sx,  sy, -sz), (-sx,  sy,  sz), ( sx,  sy,  sz)], (0, 1, 0),
        # -Y face
        [(-sx, -sy, -sz), ( sx, -sy, -sz), ( sx, -sy,  sz), (-sx, -sy,  sz)], (0, -1, 0),
        # +Z face
        [(-sx, -sy,  sz), ( sx, -sy,  sz), ( sx,  sy,  sz), (-sx,  sy,  sz)], (0, 0, 1),
        # -Z face
        [(-sx,  sy, -sz), ( sx,  sy, -sz), ( sx, -sy, -sz), (-sx, -sy, -sz)], (0, 0, -1),
    ]

    # Build vertex data: 6 faces × 4 verts each
    face_list = []
    i = 0
    while i < len(faces):
        verts = faces[i]
        n     = faces[i + 1]
        i    += 2
        face_list.append((verts, n))
        for vx, vy, vz in verts:
            vertex.addData3(vx, vy, vz)
            normal_w.addData3(*n)
            color_w.addData4(r, g, b, a)

    tris = GeomTriangles(Geom.UHStatic)
    for fi in range(6):
        base = fi * 4
        tris.addVertices(base,     base + 1, base + 2)
        tris.addVertices(base,     base + 2, base + 3)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode('box')
    node.addGeom(geom)
    return NodePath(node)


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------

def make_sphere_np(
    radius: float, color: tuple, lat_segments: int = 10, lon_segments: int = 12
) -> NodePath:
    """Create a UV sphere NodePath.

    Args:
        radius:        Sphere radius.
        color:         (r, g, b, a).
        lat_segments:  Number of horizontal rings.
        lon_segments:  Number of vertical slices.
    """
    r, g, b, a = (float(c) for c in color)

    lat = max(2, lat_segments)
    lon = max(3, lon_segments)

    n_verts = (lat + 1) * (lon + 1)
    fmt   = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData('sphere', fmt, Geom.UHStatic)
    vdata.setNumRows(n_verts)

    vertex   = GeomVertexWriter(vdata, 'vertex')
    normal_w = GeomVertexWriter(vdata, 'normal')
    color_w  = GeomVertexWriter(vdata, 'color')

    for i in range(lat + 1):
        phi = math.pi * i / lat
        sp  = math.sin(phi)
        cp  = math.cos(phi)
        for j in range(lon + 1):
            theta = 2.0 * math.pi * j / lon
            st    = math.sin(theta)
            ct    = math.cos(theta)
            nx    = sp * ct
            ny    = sp * st
            nz    = cp
            vertex.addData3(nx * radius, ny * radius, nz * radius)
            normal_w.addData3(nx, ny, nz)
            color_w.addData4(r, g, b, a)

    tris = GeomTriangles(Geom.UHStatic)
    for i in range(lat):
        for j in range(lon):
            v0 = i       * (lon + 1) + j
            v1 = i       * (lon + 1) + j + 1
            v2 = (i + 1) * (lon + 1) + j
            v3 = (i + 1) * (lon + 1) + j + 1
            tris.addVertices(v0, v2, v1)
            tris.addVertices(v1, v2, v3)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode('sphere')
    node.addGeom(geom)
    return NodePath(node)


# ---------------------------------------------------------------------------
# Cylinder (for spears)
# ---------------------------------------------------------------------------

def make_cylinder_np(
    radius: float, half_height: float, color: tuple, segments: int = 10
) -> NodePath:
    """Create a cylinder with caps.

    The cylinder's long axis runs along Z.
    """
    r, g, b, a = (float(c) for c in color)
    segs = max(3, segments)

    # Barrel + 2 cap discs
    # Barrel: 2*(segs+1) verts; top cap: segs+1; bottom cap: segs+1
    n_barrel = (segs + 1) * 2
    n_cap    = segs + 1
    total    = n_barrel + 2 * n_cap

    fmt   = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData('cylinder', fmt, Geom.UHStatic)
    vdata.setNumRows(total)

    vertex   = GeomVertexWriter(vdata, 'vertex')
    normal_w = GeomVertexWriter(vdata, 'normal')
    color_w  = GeomVertexWriter(vdata, 'color')

    # Barrel vertices
    for ring_z, nz in ((-half_height, 0.0), (half_height, 0.0)):
        for i in range(segs + 1):
            angle = 2 * math.pi * i / segs
            nx    = math.cos(angle)
            ny    = math.sin(angle)
            vertex.addData3(nx * radius, ny * radius, ring_z)
            normal_w.addData3(nx, ny, 0.0)
            color_w.addData4(r, g, b, a)

    # Top cap (z = +half_height, normal = +Z)
    for i in range(segs + 1):
        angle = 2 * math.pi * i / segs
        vertex.addData3(math.cos(angle)*radius, math.sin(angle)*radius, half_height)
        normal_w.addData3(0, 0, 1)
        color_w.addData4(r, g, b, a)

    # Bottom cap (z = -half_height, normal = -Z)
    for i in range(segs + 1):
        angle = 2 * math.pi * i / segs
        vertex.addData3(math.cos(angle)*radius, math.sin(angle)*radius, -half_height)
        normal_w.addData3(0, 0, -1)
        color_w.addData4(r, g, b, a)

    tris = GeomTriangles(Geom.UHStatic)

    # Barrel quads
    for i in range(segs):
        b0 = i
        b1 = i + 1
        t0 = (segs + 1) + i
        t1 = (segs + 1) + i + 1
        tris.addVertices(b0, t0, b1)
        tris.addVertices(b1, t0, t1)

    # Top cap fan (centre of top ring = average, but we use strip)
    top_off = n_barrel
    for i in range(segs - 1):
        tris.addVertices(top_off, top_off + i + 1, top_off + i + 2)

    # Bottom cap fan
    bot_off = n_barrel + n_cap
    for i in range(segs - 1):
        tris.addVertices(bot_off, bot_off + i + 2, bot_off + i + 1)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode('cylinder')
    node.addGeom(geom)
    return NodePath(node)


# ---------------------------------------------------------------------------
# Terrain mesh
# ---------------------------------------------------------------------------

def build_terrain_mesh(
    heightmap: np.ndarray, world_size: float
) -> NodePath:
    """Build a Panda3D triangle mesh from a 2-D heightmap.

    Args:
        heightmap:  (rows, cols) float32 array of heights in metres.
        world_size: World width/depth in metres.

    Returns:
        NodePath for the terrain mesh (not yet parented).
    """
    rows, cols = heightmap.shape
    dx = world_size / (cols - 1)
    dy = world_size / (rows - 1)
    half = world_size / 2

    n_verts = rows * cols
    fmt   = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData('terrain', fmt, Geom.UHStatic)
    vdata.setNumRows(n_verts)

    vertex   = GeomVertexWriter(vdata, 'vertex')
    normal_w = GeomVertexWriter(vdata, 'normal')
    color_w  = GeomVertexWriter(vdata, 'color')

    # Colour based on height: low → brown, high → green
    h_max = float(heightmap.max()) if heightmap.max() > 0 else 1.0
    for i in range(rows):
        for j in range(cols):
            x = j * dx - half
            y = i * dy - half
            z = float(heightmap[i, j])
            vertex.addData3(x, y, z)

            # Approximate normal via finite differences
            dzdx = (
                float(heightmap[i, min(j+1, cols-1)]) -
                float(heightmap[i, max(j-1, 0)])
            ) / (2 * dx)
            dzdy = (
                float(heightmap[min(i+1, rows-1), j]) -
                float(heightmap[max(i-1, 0), j])
            ) / (2 * dy)
            nx, ny, nz = -dzdx, -dzdy, 1.0
            length = math.sqrt(nx*nx + ny*ny + nz*nz)
            normal_w.addData3(nx/length, ny/length, nz/length)

            # Colour: brown at low, green at high
            t = z / h_max
            cr = 0.40 - 0.30 * t
            cg = 0.26 + 0.28 * t
            cb = 0.13 - 0.05 * t
            color_w.addData4(cr, cg, cb, 1.0)

    tris = GeomTriangles(Geom.UHStatic)
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = i * cols + j
            v1 = i * cols + j + 1
            v2 = (i + 1) * cols + j
            v3 = (i + 1) * cols + j + 1
            tris.addVertices(v0, v1, v2)
            tris.addVertices(v1, v3, v2)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode('terrain_mesh')
    node.addGeom(geom)
    return NodePath(node)
