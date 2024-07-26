import numpy as np

from pyvista import CellType


# Definition of number of nodes for element type
CELL_NODE_NUMBERS = {
    CellType.VERTEX: 1,
    CellType.LINE: 2,
    CellType.TRIANGLE: 3,
    CellType.QUAD: 4,
    CellType.TETRA: 4,
    CellType.PYRAMID: 5,
    CellType.WEDGE: 6,
    CellType.HEXAHEDRON: 8,
    None: 10,
}

# Cell type mapping to PFLOTRAN implicit unstructured grid
PF_CELL_TYPE_MAP = np.array(
    [
        None,  # vertex
        None,  # line
        None,  # triangle
        None,  # quad
        "T",  # tetra
        "P",  # pyramid
        "W",  # prism
        "H",  # hexahedron
        None,  # hybrid
        None,  # polygon
    ]
)

# Cell type mapping to VTK cell
VTK_CELL_TYPE_MAP = np.array(
    [
        CellType.VERTEX,
        CellType.LINE,
        CellType.TRIANGLE,
        CellType.QUAD,
        CellType.TETRA,
        CellType.PYRAMID,
        CellType.WEDGE,  # prism, VTK order = [0, 2, 1, 3, 5, 4]
        CellType.HEXAHEDRON,
        None,  # hybrid, no corresponding type
        None,  # polygon, no corresponding type
    ]
)
