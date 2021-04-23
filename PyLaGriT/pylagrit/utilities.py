import os
import numpy as np
from datetime import datetime


def zone_to_zonn(zonefile):
    zonnfile = os.path.splitext(zonefile)[0] + ".zonn"
    with open(zonefile, "r") as fh:
        fh.readline()
        lns = fh.readlines()
    with open(zonnfile, "w") as fout:
        fout.write("zonn\n")
        for l in lns:
            fout.write(l)


def spherical_writeFEHM(node_locations, filename_base, title="default"):
    """
    Create FEHM .stor and .fehmn input files for 1d spherical simulations.
    Assumes logical 1D structure.
    :arg node_locations: radial location of nodes
    :type node_locations: array(float)
    :arg filename_base: base name for the .stor and .fehmn files
    :type filename_base: str
    :arg title: optional title for simulation
    :type title: str

    Example:
    from pylagrit import utilities as util
    import numpy as np
    nodes = np.linspace(0.0,200.0)
    filename_base = "test"
    title = "test sim"
    # write .stor and .fehmn files
    util.spherical_writeFEHM(nodes,filename_base,title)
    """
    # stor file header
    now = datetime.now()
    print_datetime = now.strftime("%m/%d/%Y  %H:%M:%S")
    header = (
        "FEHM .stor file generated by PyLaGriT  "
        + print_datetime
        + "\n"
        + "title:  "
        + title
        + "\n"
    )

    # calculate dx, face locations, face areas, volumes, and geometric coefficients
    dx = spherical_dx(node_locations)
    faces = spherical_faces(node_locations)
    area = spherical_areas(faces)
    volumes = spherical_volumes(node_locations)
    coeffs = area / dx

    # matrix metadata header info
    neq = np.size(node_locations)
    num_coeffs = neq - 1
    FEHM_mem = (neq - 2) * 3 + 4 + neq + 1
    num_area_coeff = 1
    max_connect = 3
    matrix_params = (num_coeffs, neq, FEHM_mem, num_area_coeff, max_connect)
    params_header = "        " + "        ".join(str(s) for s in matrix_params) + "\n"

    # row count
    row_count = np.empty([neq + 1])
    start = neq + 1
    row_count[0] = start
    row_count[1] = row_count[0] + 2
    for i in range(2, neq):
        row_count[i] = row_count[i - 1] + 3
    row_count[neq] = row_count[neq - 1] + 2

    # row entries
    row_entries = []
    row_entries.append(1)
    row_entries.append(2)
    for i in range(1, neq - 1):
        row_entries.append(i)
        row_entries.append(i + 1)
        row_entries.append(i + 2)
    row_entries.append(neq - 1)
    row_entries.append(neq)

    # index into geometric coefficient matrix
    coeff_indices = []
    coeff_indices.append(0)
    coeff_indices.append(1)
    for i in range(1, neq - 1):
        coeff_indices.append(i)
        coeff_indices.append(0)
        coeff_indices.append(i + 1)
    coeff_indices.append(neq - 1)
    coeff_indices.append(0)
    # neq + 1 extra 0s for padding
    for i in range(0, neq + 1):
        coeff_indices.append(0)

    # index of diagonal entries
    diagonal_indices = np.empty([neq])
    diagonal_indices[0] = neq + 2
    for i in range(1, neq):
        diagonal_indices[i] = diagonal_indices[i - 1] + 3

    # open stor file, write header and matrix parameters
    storfile = filename_base + ".stor"
    sfile = open(storfile, "w")
    _ = sfile.write(header)
    _ = sfile.write(params_header)

    # write Voronoi volumes
    count = 1
    for vol in volumes:
        print("  %1.12e" % vol, end="" if count % 5 else "\n", file=sfile)
        count += 1
    if (count - 1) % 5:
        _ = sfile.write("\n")

    # write row counts
    count = 1
    for row in row_count:
        print(" %9d" % row, end="" if count % 5 else "\n", file=sfile)
        count += 1
    if (count - 1) % 5:
        _ = sfile.write("\n")

    # write row entries
    count = 1
    for row in row_entries:
        print(" %9d" % row, end="" if count % 5 else "\n", file=sfile)
        count += 1
    if (count - 1) % 5:
        _ = sfile.write("\n")

    # write geometric coefficient indices
    count = 1
    for idx in coeff_indices:
        print(" %9d" % idx, end="" if count % 5 else "\n", file=sfile)
        count += 1
    if (count - 1) % 5:
        _ = sfile.write("\n")

    # write diagonal indices
    count = 1
    for idx in diagonal_indices:
        print(" %9d" % idx, end="" if count % 5 else "\n", file=sfile)
        count += 1
    if (count - 1) % 5:
        _ = sfile.write("\n")

    # write geometric coefficients
    count = 1
    for coef in coeffs:
        print("  %1.12e" % coef, end="" if count % 5 else "\n", file=sfile)
        count += 1

    # close stor file
    if (count - 1) % 5:
        _ = sfile.write("\n")
    _ = sfile.close()

    # open fehmn file
    fehmnfile = filename_base + ".fehmn"
    ifile = open(fehmnfile, "w")
    # write header and neq
    _ = ifile.write("coor\n%d\n" % neq)
    # write node number and radial location
    for i in range(1, neq + 1):
        print(
            "        %3d        %12f        0        0" % (i, node_locations[i - 1]),
            file=ifile,
        )
    # write connectivity
    _ = ifile.write("\nelem\n")
    _ = ifile.write("%d %d\n" % (2, neq - 1))
    for i in range(1, neq):
        print("%3d   %3d   %3d" % (i, i, i + 1), file=ifile)
    # close fehmn file
    _ = ifile.write("\nstop\n")
    _ = ifile.close()


def spherical_faces(node_locations):
    """
    Calculate radial interface locations given radial node locations.
    Assumes 1st and last nodes lie on domain boundary.
    :arg node_locations: radial location of nodes
    :type node_locations: array_like(float)
    Returns: array of radial interface locations of size node_locations - 1
    """
    interface_locations = node_locations[:-1] + np.diff(node_locations) / 2.0
    return interface_locations


def spherical_areas(interface_locations):
    """
    Calculate spherical area of each interface given radial interface locations.
    :arg interface_locations: radial location of interfaces
    :type interface_locations: array_like(float)
    Returns: array of spherical interface areas of size interface_locations
    """
    areas = 4.0 * np.pi * interface_locations ** 2
    return areas


def spherical_dx(node_locations):
    """
    Calculate Delaunay edge lengths given radial node locations.
    :arg node_locations: radial location of nodes
    :type node_locations: array_like(float)
    Returns: array of Delaunay edge lengths of size node_locations - 1
    """
    delaunay = np.diff(node_locations)
    return delaunay


def spherical_volumes(node_locations):
    """
    Calculate Voronoi volume associated with each node given radial node locations.
    :arg node_locations: radial location of nodes
    :type node_locations: array_like(float)
    Returns: array of Voronoi volumes of size node_locations
    """
    coeff = np.pi * 4.0 / 3.0
    edges = spherical_faces(node_locations)
    edges = np.insert(edges, 0, node_locations[0])
    edges = np.append(edges, node_locations[np.size(node_locations) - 1])
    spheres = coeff * edges ** 3
    volumes = np.diff(spheres)
    assert np.all(volumes > 0), "ERROR: Negative volumes are not good."
    return volumes
