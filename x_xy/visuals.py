import numpy as np
from vispy.geometry.meshdata import MeshData
from vispy.scene.visuals import Mesh, create_visual_node
from vispy.visuals import CompoundVisual, TubeVisual

from x_xy.base import Color


class DoubleMeshVisual(CompoundVisual):
    _lines: Mesh
    _faces: Mesh

    def __init__(
        self, verts, edges, faces, *, color: Color = None, edge_color: Color = None
    ):
        # if color is None:
        #     color = (0.5, 0.5, 1, 1)

        if color is not None:
            self._faces = Mesh(verts, faces, color=color, shading="smooth")
            self.light_dir = np.array([0, -1, 0])
        else:
            self._faces = Mesh()

        if edge_color is not None:
            self._edges = Mesh(verts, edges, color=edge_color, mode="lines")
        else:
            self._edges = Mesh()

        super().__init__([self._faces, self._edges])
        self._faces.set_gl_state(
            polygon_offset_fill=True, polygon_offset=(1, 1), depth_test=True
        )


def box_mesh(
    dim_x: float, dim_y: float, dim_z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts = np.array(
        [
            (-dim_x, -dim_y, -dim_z),
            (dim_x, -dim_y, -dim_z),
            (-dim_x, dim_y, -dim_z),
            (dim_x, dim_y, -dim_z),
            (-dim_x, -dim_y, dim_z),
            (dim_x, -dim_y, dim_z),
            (-dim_x, dim_y, dim_z),
            (dim_x, dim_y, dim_z),
        ],
        dtype=np.float32,
    )

    verts /= 2

    edges = np.array(
        [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ],
        dtype=np.uint32,
    )

    faces = np.array(
        [
            (0, 1, 2),
            (1, 2, 3),
            (0, 1, 4),
            (1, 4, 5),
            (0, 2, 4),
            (2, 4, 6),
            (1, 3, 5),
            (3, 5, 7),
            (2, 3, 6),
            (3, 6, 7),
            (4, 5, 6),
            (5, 6, 7),
        ]
    )

    return verts, edges, faces


class BoxVisual(DoubleMeshVisual):
    # NOTE: need a custom BoxVisual class, since vispy.scene.visuals.Box does not
    # support shading

    def __init__(
        self,
        dim_x: float,
        dim_y: float,
        dim_z: float,
        *,
        color: Color = None,
        edge_color: Color = None
    ):
        dim_x = float(dim_x)
        dim_y = float(dim_y)
        dim_z = float(dim_z)

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        verts, edges, faces = box_mesh(dim_x, dim_y, dim_z)

        super().__init__(verts, edges, faces, color=color, edge_color=edge_color)


Box = create_visual_node(BoxVisual)


class CylinderVisual(TubeVisual):
    def __init__(
        self,
        radius: float,
        length: float,
        *,
        color: Color = None,
        edge_color: Color = None
    ):
        radius = float(radius)
        length = float(length)

        num_points = min(100 * int(length), 100)

        points = np.zeros((num_points, 3))
        points[:, 0] = np.linspace(-length / 2, length / 2, num_points)

        self.radius = radius
        self.length = length

        super().__init__(
            points,
            radius,
            closed=True,
            color=color,
            shading="smooth",
        )


Cylinder = create_visual_node(CylinderVisual)


def capsule_mesh(radius: float, length: float, offset: bool = True) -> MeshData:
    if length < 2 * radius:
        raise ValueError("length must be at least 2 * radius")

    sphere_rows = max(8 * int(radius), 5)

    cols = 8

    cyl_length = length - 2 * radius
    cyl_rows = max(4 * int(cyl_length), 3)

    total_rows = 2 * sphere_rows + cyl_rows

    verts = np.empty((total_rows, cols, 3), dtype=np.float32)

    # compute vertices
    phi = np.linspace(0.0, np.pi, 2 * sphere_rows)

    verts[:sphere_rows, :, 0] = (
        radius * np.cos(phi[:sphere_rows, None]) + cyl_length / 2
    )
    verts[sphere_rows:-sphere_rows, :, 0] = np.linspace(
        -cyl_length / 2, cyl_length / 2, cyl_rows
    )[::-1, None]
    verts[-sphere_rows:, :, 0] = (
        radius * np.cos(phi[-sphere_rows:, None]) - cyl_length / 2
    )

    th = (np.linspace(0, 2 * np.pi, cols))[None, :]

    if offset:
        # rotate each row by 1/2 column
        th = th + (np.pi / cols) * np.arange(total_rows)[:, None]

    verts[..., 1] = radius * np.cos(th)
    verts[..., 2] = radius * np.sin(th)

    verts[:sphere_rows, :, 1:3] *= np.sin(phi[:sphere_rows, None, None])
    verts[-sphere_rows:, :, 1:3] *= np.sin(phi[-sphere_rows:, None, None])

    verts = verts.reshape(-1, 3)

    # compute faces
    faces = np.empty(((total_rows - 1) * cols * 2, 3), dtype=np.uint32)

    rowtemplate1 = (
        (np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 0]])) % cols
    ) + np.array([[cols, 0, 0]])

    rowtemplate2 = (
        (np.arange(cols).reshape(cols, 1) + np.array([[1, 1, 0]])) % cols
    ) + np.array([[cols, 0, cols]])

    for row in range(total_rows - 1):
        start = row * cols * 2

        faces[start : start + cols] = rowtemplate1 + row * cols
        faces[start + cols : start + (cols * 2)] = rowtemplate2 + row * cols

    mesh = MeshData(vertices=verts, faces=faces)

    return mesh.get_vertices(), mesh.get_edges(), mesh.get_faces()


class CapsuleVisual(DoubleMeshVisual):
    def __init__(
        self,
        radius: float,
        length: float,
        *,
        color: Color = None,
        edge_color: Color = None
    ):
        radius = float(radius)
        length = float(length)

        self.radius = radius
        self.length = length

        verts, edges, faces = capsule_mesh(radius, length)

        super().__init__(verts, edges, faces, color=color, edge_color=edge_color)


Capsule = create_visual_node(CapsuleVisual)
