import os
import warnings
import xml.etree.ElementTree as ET

from collections import OrderedDict
from itertools import product
from pathlib import Path
from subprocess import call
from typing import Dict, List, Literal, Optional, Tuple, cast
from xml.dom import minidom

import numpy

from pexpect import spawn

from .utilities import decode_binary, make_name, minus_self


catch_errors = True


class LaGriT_Warning(Warning):
    pass


class PyLaGriT(spawn):
    """
    Python lagrit class

    :param lagrit_exe: Path to LaGriT executable
    :type lagrit_exe: str
    :param verbose: If True, LaGriT terminal output will be displayed
    :type verbose: bool
    :param batch: If True, PyLaGriT will be run in batch mode, collecting LaGriT commands until the run_batch method is called.
    :type batch: bool
    :param batchfile: Name of batch file to use if batch is True
    :type batchfile: str
    :param gmv_exe: Path to GMV executable
    :type gmv_exe: str
    :param paraview_exe: Path to ParaView executable
    :type paraview_exe: str
    :param timeout: Number of seconds to wait for response from LaGriT
    """

    def __init__(
        self,
        lagrit_exe: Optional[str] = None,
        verbose=True,
        batch=False,
        batchfile="pylagrit.lgi",
        gmv_exe: Optional[str] = None,
        paraview_exe: Optional[str] = None,
        timeout=300,
        **kwargs,
    ):
        self.verbose = verbose
        self.mo: Dict[str, MO] = {}
        self.pset: Dict[str, PSet] = {}
        self.batch = batch
        self._check_rc()

        if lagrit_exe is not None:
            self.lagrit_exe = lagrit_exe

        if self.lagrit_exe is None or not os.path.exists(self.lagrit_exe):
            raise FileNotFoundError(
                "Error: LaGriT executable is not defined. Add 'lagrit_exe' "
                "option to PyLaGriT (e.g., lg = pylagrit.PyLaGriT(lagrit_exe"
                "=<path/to/lagrit/exe>), or create a pylagritrc file as "
                "described in the manual."
            )

        if gmv_exe is not None:
            self.gmv_exe = gmv_exe
        if paraview_exe is not None:
            self.paraview_exe = paraview_exe
        if self.batch:
            try:
                self.fh = open(batchfile, "w")
            except OSError:
                print("Unable to open " + batchfile)
                print("Batch mode disabled")
                self.batch = False
            else:
                self.batchfile = batchfile
                self.fh.write("# PyLaGriT generated LaGriT script\n")
        else:
            kwargs["timeout"] = timeout
            super().__init__(self.lagrit_exe, **kwargs)
            self.expect("Enter a command")
            if verbose:
                print(decode_binary(self.before))

    def run_batch(self):
        self.fh.write("finish\n")
        self.fh.close()
        if self.verbose:
            call(self.lagrit_exe + " < " + self.batchfile, shell=True)  # noqa: S602
        else:
            fout = open("pylagrit.stdout", "w")
            call(self.lagrit_exe + " < " + self.batchfile, shell=True, stdout=fout)  # noqa: S602
            fout.close()

    def sendcmd(self, s: str, verbose=True, expectstr="Enter a command"):
        if self.batch:
            self.fh.write(s + "\n")
        else:
            super().sendline(s)
            self.expect(expectstr)
            if verbose and self.verbose:
                print(decode_binary(self.before))

            if catch_errors:
                for _line in decode_binary(self.before).split("\n"):
                    if "ERROR" in _line:
                        raise Exception(_line)
                    elif "WARNING" in _line:
                        warnings.warn(_line, category=LaGriT_Warning, stacklevel=2)

    def interact(self, escape_character="^", input_filter=None, output_filter=None):
        if self.batch:
            print("Interactive mode unavailable during batch mode")
        else:
            print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Entering interactive mode")
            print(
                "To return to python terminal, type a '"
                + escape_character
                + "' character"
            )
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
            print(self.after)
            super().interact(
                escape_character=escape_character,
                input_filter=input_filter,
                output_filter=output_filter,
            )

    def cmo_status(self, cmo: "Optional[MO | str]" = None, brief=False, verbose=True):
        cmd = "cmo/status"
        if cmo:
            cmd += "/" + str(cmo)
        if brief:
            cmd += "/brief"
        self.sendcmd(cmd, verbose=verbose)

    def read_mo(
        self,
        filename: str,
        filetype: Optional[str] = None,
        name: Optional[str] = None,
        binary=False,
    ) -> Optional["MO" | List["MO"]]:
        """
        Read in mesh

        :param filename: Name of mesh file to read in
        :type filename: str
        :param filetype: Type of file, automatically detected if not specified
        :type filetype: str
        :param name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str
        :param binary: Indicates that file is binary if True, ascii if False
        :type binary: bool
        :returns: MO

        Example 1:
            >>> # To use pylagrit, import the module.
            >>> import pylagrit
            >>> # Create your pylagrit session.
            >>> lg = pylagrit.PyLaGriT()
            >>> # Create a mesh object and dump it to a gmv file 'test.gmv'.
            >>> mo = lg.create(name="test")
            >>> mo.createpts_brick_xyz((5, 5, 5), (0, 0, 0), (5, 5, 5))
            >>> mo.dump("test.gmv")
            >>> mo.dump("test.avs")
            >>> mo.dump("test.lg")
            >>> mo1 = lg.read_mo("test.gmv")
            >>> mo2 = lg.read_mo("test.avs")
            >>> mo3 = lg.read_mo("test.lg", name="test")

        Example 2 - Reading in LaGriT binary file, autodetect mesh object name
            >>> # To use pylagrit, import the module.
            >>> import pylagrit
            >>> import numpy
            >>> # Instantiate the lagrit object.
            >>> lg = pylagrit.PyLaGriT()
            >>> # Create list with mesh object as first element
            >>> dxyz = numpy.array([0.25] * 3)
            >>> mins = numpy.array([0.0] * 3)
            >>> maxs = numpy.array([1.0] * 3)
            >>> ms = [
            ...     lg.createpts_dxyz(dxyz, mins, maxs, "tet", connect=True, name="testmo")
            ... ]
            >>> # Create three new mesh objects, each one directly above the other
            >>> for i in range(3):
            >>>     ms.append(ms[-1].copy())
            >>>     ms[-1].trans(ms[-1].mins,ms[-1].mins+numpy.array([0.,0.,1.]))
            >>> lg.dump("lagrit_binary.lg")
            >>> lg.close()
            >>> lg = pylagrit.PyLaGriT()
            >>> ms_read = lg.read_mo("lagrit_binary.lg")
            >>> print 'Name of mesh object read in should be testmo, is: ', ms_read.name
        """

        # If filetype is lagrit, name is irrelevant
        islg = filetype == "lagrit" or filename.split(".")[-1] in [
            "lg",
            "lagrit",
            "LaGriT",
        ]

        cmd = ["read", filename]

        if filetype is not None:
            cmd.append(filetype)

        if islg:
            cmd.append("dum")
        else:
            if name is None:
                name = make_name("mo", self.mo.keys())
            cmd.append(name)

        if binary:
            cmd.append("binary")

        self.sendcmd("/".join(cmd))

        # If format lagrit, cmo read in will not be set to name
        if islg and not self.batch:
            self.sendcmd("cmo/status/brief", verbose=False)
            # dump lagrit doesn't seem to ever dump multiple mos now???
            mos = []
            for line in decode_binary(self.before).splitlines():
                if "Mesh Object name:" in line:
                    nm = line.split(":")[1].strip()
                    self.mo[nm] = MO(nm, self)
                    mos.append(self.mo[nm])
            if len(mos) == 1:
                if name is not None and name != mos[0].name:
                    self.sendcmd("cmo/copy/" + name + "/" + mos[0].name)
                    self.sendcmd("cmo/release/" + mos[0].name)
                return mos[0]
            elif len(mos) > 1:
                if name is not None:
                    print("Multiple mesh objects exist, 'name' option will be ignored")
                return mos
        else:
            if name is None:
                name = make_name("mo", self.mo.keys())
            self.mo[name] = MO(name, self)
            return self.mo[name]

    def read_fehm(
        self, filename: str, avs_filename="temp.inp", elem_type: Optional[str] = None
    ):
        with open(filename) as fh:
            ln = fh.readline()
            nn = int(fh.readline().strip())
            while "elem" not in ln:
                ln = fh.readline()
            vs = fh.readline().strip().split()
        elem_int = int(vs[0])
        ne = int(vs[1])
        crds = numpy.genfromtxt(filename, skip_header=2, max_rows=nn)
        conns = numpy.genfromtxt(filename, skip_header=2 + nn + 3, max_rows=ne)
        with open(avs_filename, "w") as fh:
            fh.write("    %d    %d    0    0    0\n" % (nn, ne))
            numpy.savetxt(fh, crds, fmt="%d %f %f %f")
            if elem_type is None:
                if elem_int == 8:
                    elem_type = "hex"
                elif elem_int == 3:
                    elem_type = "tri"
                elif elem_int == 4:
                    if (
                        numpy.all(numpy.diff(crds[:, 1])) == 0
                        or numpy.all(numpy.diff(crds[:, 2])) == 0
                        or numpy.all(numpy.diff(crds[:, 3])) == 0
                    ):
                        elem_type = "quad"
                    else:
                        elem_type = "tet"
            for conn in conns:
                fh.write("%d 1 %s" % (conn[0], elem_type))
                for i in range(elem_int):
                    fh.write(" %d" % conn[i + 1])
                fh.write("\n")
        return self.read_mo(avs_filename)

    def read_sheetij(
        self,
        name: str,
        filename: str,
        NXY: Tuple[int, int],
        minXY: Tuple[float, float],
        DXY: Tuple[float, float],
        connect=True,
        file_type="ascii",
        flip="none",
        skip_lines=0,
        data_type="float",
    ):
        """
        Creates a quad mesh from an elevation file. Note the input file is read as Z(i,j) into the cmo attribute 'zic'

        :param name: name of mesh object
        :type name: string
        :param filename: Elevation filename
        :type filename: string
        :param NXY: [nx, ny] - [columns in x-direction, rows in y-direction]
        :type NXY: list
        :param minXY: [minX, minY] - location of lower left corner
        :type minXY: list
        :param DXY: [Dx, Dy] - cell size in x and y directions
        :type DXY: list
        :param connect: True will create a quad grid, otherwise keeps data as points
        :type connect: bool
        :param file_type: May be either ascii or binary
        :type file_type: string
        :param flip: May be 'x', 'y' to reflect across those axes, or 'none' to keep static
        :type flip: string
        :param skip_lines: skip n number of header lines
        :type skip_lines: integer
        :param data_type: read in elevation data as either float or double
        :type data_type: string
        :returns: MO

        Example 1 - Building a surface mesh from Modflow elevation file:
            >>> # To use pylagrit, import the module.
            >>> from pylagrit import PyLaGriT
            >>> import numpy as np
            >>>
            >>> # Instantiate PyLaGriT
            >>> l = PyLaGriT()
            >>>
            >>> # Elevation files are typically headerless unwrapped vectors
            >>> # Define parameters to pack these elements into a matrix
            >>> ncols = 276
            >>> nrows = 313
            >>> DXY = [100, 100]
            >>>
            >>> elev_surface = l.read_sheetij(
            ...     "surfacemesh", "example.mod", [ncols, nrows], [0, 0], DXY, flip="y"
            ... )
            >>> elev_surface.paraview()

        """

        connect_str = "connect" if connect else "points"
        skip_str = "skip %d" % skip_lines

        data_type = data_type.lower()
        file_type = file_type.lower()

        if data_type not in ["float", "double"]:
            raise ValueError("data_type must be float or double")

        if file_type not in ["ascii", "binary"]:
            raise ValueError("file_type must be ascii or binary")

        flip_str = flip.lower()

        if flip_str in ["x", "y", "xy", "none"]:
            if flip_str == "x":
                flip_str = "xflip"
            if flip_str == "y":
                flip_str = "yflip"
            if flip_str == "xy":
                flip_str = "xflip,yflip"
            if flip_str == "none":
                flip_str = ""
        else:
            raise ValueError("Argument flip must be: 'x', 'y', 'xy', or 'none'")

        # Create new mesh object with given name
        self.sendcmd(f"cmo/create/{name}")
        self.sendcmd(f"cmo/select/{name}")

        # Read in elevation file and append to mesh
        cmd = [
            "read",
            "sheetij",
            filename,
            ",".join([str(v) for v in NXY]),
            ",".join([str(v) for v in minXY]),
            ",".join([str(v) for v in DXY]),
            skip_str,
            flip_str,
            connect_str,
            file_type,
            data_type,
        ]
        self.sendcmd("/".join(cmd))

        self.mo[name] = MO(name, self)
        return self.mo[name]

    def read_modflow(
        self,
        materials_file: str,
        nrows: int,
        ncols: int,
        name: Optional[str] = None,
        DXY=(100, 100),
        height=7.75,
        filename: Optional[str] = None,
    ):
        """
        Reads in a Modflow elevation file (and, optionally, an HDF5/txt file containing node materials) and generates and returns hexagonal mesh.

        :param filename: Filename of Modflow elevation data file
        :type filename: str
        :param nrows: Number of rows in elevation file
        :type nrows: int
        :param ncols: Number of columns in elevation file
        :type ncols: int
        :param name: Name of returned mesh (optional)
        :type name: str
        :param DXY: Spacing in x/y directions
        :type DXY: list (number)
        :param height: The 'thickness' in the Z-direction of the returned hex mesh
        :type height: float
        :param materials_file: A text or HDF5 binary file containing materials properties for an elevation mesh
        :type materials_file: str
        :param materials_keys: A list containing the keys to the materials array, ordered sequentially. If set, it is assumed materials_file is an HDF5 file.
        :type materials_keys: list (str)
        :returns: MO
        """

        if name is None:
            name = make_name("mo", self.mo.keys())

        x = numpy.arange(0, ncols + 1, 1)
        y = numpy.arange(0, nrows + 1, 1)
        z = numpy.arange(
            0, 2 * height, height
        )  # x2 because of half-open interval: [start, stop)

        # Generate hexmesh
        # Alternately, just extrude elev_surface
        hexmesh = self.gridder(
            x.tolist(), y.tolist(), z.tolist(), elem_type="hex", connect=True, name=name
        )

        # Capture hexmesh points as pset
        hexset = hexmesh.pset_geom(
            (0, 0, 0),
            (max(x), max(y), max(z)),
            ctr=(0, 0, 0),
            stride=(0, 0, 0),
            geom="xyz",
            name="hexset",
        )

        # Scale hexmesh to match length of surface (optimize later)
        hexset.scale("relative", "xyz", [DXY[0], DXY[1], 1], [0, 0, 0])

        # Translate such that 50% of mesh is above z=0 and 50% is under
        hexset.trans((0, 0, 0), (0, 0, -height / 2))

        # Capture points < 0
        hex_bottom = hexmesh.pset_attribute(
            "zic", 0, comparison="lt", stride=(0, 0, 0), name="pbot"
        )

        # Set hex mesh z-coord to 0
        hexmesh.setatt("zic", 0.0)

        imt_data = numpy.loadtxt(materials_file)

        # Write out to hidden materials file
        tmp_file = "._tmp_materials.txt"
        tmp_materials = open(tmp_file, "w")

        imt_dims = numpy.shape(imt_data)
        nrows = imt_dims[0]
        ncols = imt_dims[1]

        # imt_types = numpy.unique(imt_data).tolist()

        # Ensure that imt values are greater than 0
        # imt_min = min(imt_types)
        correction = 0

        # if imt_min < 0:
        #    imt_types = [int(i + 1 + abs(imt_min)) for i in imt_types]
        #    correction = 1 + abs(imt_min)
        # elif imt_min == 0:
        #    imt_types = [int(i + 1) for i in imt_types]
        #    correction = 1

        # Unpack matrix into vector and write
        for i in range(0, nrows):
            for j in range(0, ncols):
                imt_value = int(imt_data[(nrows - 1) - i][j]) + correction
                tmp_materials.write(f"{imt_value}\n")

        # Close write file
        tmp_materials.close()

        # Project materials onto surface
        mtrl_surface = self.read_sheetij(
            "mo_mat", tmp_file, (ncols, nrows), (0, 0), DXY
        )

        # Create psets based on imt values, assign global imt from psets
        # for i in range(0,len(imt_types)):
        #    mtrl_surface.pset_attribute('zic', imt_types[i], comparison='eq', stride=(0,0,0), name='p{}'.format(i))
        #    mtrl_surface.setatt('imt', imt_types[i], stride=['pset','get','p{}'.format(i)])

        # mtrl_surface.setatt('zic', 0.)

        hexmesh.addatt("mod_bnds", vtype="VINT", rank="scalar", length="nelements")
        hexmesh.copyatt("zic", attname_sink="mod_bnds", mo_src=mtrl_surface)
        self.sendcmd(f"cmo/printatt/{hexmesh.name}/mod_bnds/minmax")
        self.sendcmd(f"cmo/printatt/{mtrl_surface.name}/zic/minmax")

        hexmesh.addatt("pts_topbot")
        hexmesh.setatt("pts_topbot", 1.0)
        hexmesh.setatt("pts_topbot", 2.0, stride=["pset", "get", hex_bottom.name])

        # hexmesh.addatt('newimt')
        # hexmesh.interpolate('continuous','newimt',mtrl_surface,'imt')
        # hexmesh.copyatt('newimt','imt') # Probably unnecessary
        # hexmesh.delatt('newimt')

        if filename is not None:
            # Load modflow elevation map into surface
            elev_surface = self.read_sheetij(
                "motmp", filename, (ncols, nrows), (0, 0), DXY, flip="y"
            )

            # Copy elevation to new attribute and set all surface point height to 0
            elev_surface.addatt("z_elev")
            elev_surface.copyatt("zic", "z_elev", elev_surface)
            elev_surface.setatt("zic", 0.0)

            # Interpolate elevation onto z_new, copy z_new to Z, translate the bottom half of pts to fill out mesh
            hexmesh.addatt("z_new")
            hexmesh.interpolate("continuous", "z_new", elev_surface, "z_elev")
            hexmesh.copyatt("z_new", "zic")
            hexmesh.math(
                "add",
                "zic",
                value=-height,
                stride=["pset", "get", hex_bottom.name],
                attsrc="z_new",
            )
            hexmesh.delatt("z_new")
        else:
            hexmesh.math(
                "add",
                "zic",
                value=height,
                stride=["pset", "get", hex_bottom.name],
                attsrc="zic",
            )

        self.mo[name] = MO(name, self)
        return self.mo[name]

    def boundary_components(
        self,
        style="node",
        material_id_number: Optional[int] = None,
        reset: Optional[bool] = None,
    ):
        """
        Calculates the number of connected components of a mesh for diagnostic purposes.

        :param style: May be element or node
        :type style: string
        :param material_id_number: Only examines nodes with imt = mat. id number
        :type material_id_number: int
        :param reset: May be either True, False, or None
        :type reset: bool
        """

        cmd = ["boundary_components", style]

        if material_id_number:
            cmd.append(str(material_id_number))
        if reset is not None:
            if reset:
                cmd.append("reset")
            elif not reset:
                cmd.append("noreset")

        self.sendcmd("/".join(cmd))

    def addmesh(
        self,
        mo1: "MO | str",
        mo2: "MO | str",
        style="add",
        name: Optional[str] = None,
        *args,
    ):
        if name is None:
            name = make_name("mo", self.mo.keys())
        cmd = "/".join(["addmesh", style, name, str(mo1), str(mo2)])
        for a in args:
            if isinstance(a, str):
                cmd = "/".join([cmd, a])
            elif isinstance(a, list):
                cmd = "/".join([cmd, " ".join([str(v) for v in a])])
        self.sendcmd(cmd)
        self.mo[name] = MO(name, self)
        return self.mo[name]

    def addmesh_add(
        self,
        mo1: "MO | str",
        mo2: "MO | str",
        name: Optional[str] = None,
        refine_factor=-1,
        refine_style="edge",
    ):
        return self.addmesh(mo1, mo2, "add", name, refine_factor, refine_style)

    def addmesh_amr(self, mo1: "MO | str", mo2: "MO | str", name: Optional[str] = None):
        return self.addmesh(mo1, mo2, style="amr", name=name)

    def addmesh_append(
        self, mo1: "MO | str", mo2: "MO | str", name: Optional[str] = None
    ):
        return self.addmesh(mo1, mo2, style="append", name=name)

    def addmesh_delete(
        self, mo1: "MO | str", mo2: "MO | str", name: Optional[str] = None
    ):
        return self.addmesh(mo1, mo2, style="delete", name=name)

    def addmesh_glue(
        self, mo1: "MO | str", mo2: "MO | str", name: Optional[str] = None
    ):
        return self.addmesh(mo1, mo2, style="glue", name=name)

    def addmesh_intersect(
        self,
        pset: "PSet | str",
        mo1: "MO | str",
        mo2: "MO | str",
        name: Optional[str] = None,
    ):
        if name is None:
            name = make_name("mo", self.mo.keys())
        cmd = "/".join(["addmesh", "intersect", name, str(pset), str(mo1), str(mo2)])
        self.sendcmd(cmd)
        self.pset[name] = PSet(name, self)
        return self.pset[name]

    def addmesh_merge(
        self, mo1: "MO | str", mo2: "MO | str", name: Optional[str] = None
    ):
        return self.addmesh(mo1, mo2, style="merge", name=name)

    def addmesh_pyramid(
        self, mo1: "MO | str", mo2: "MO | str", name: Optional[str] = None
    ):
        return self.addmesh(mo1, mo2, style="pyramid", name=name)

    def addmesh_excavate(
        self,
        mo1: "MO | str",
        mo2: "MO | str",
        name: Optional[str] = None,
        bfs=False,
        connect=False,
    ):
        if bfs:
            bfsstr = "bfs"
        else:
            bfsstr = " "
        if connect:
            connectstr = "connect"
        else:
            connectstr = " "
        return self.addmesh(mo1, mo2, "excavate", name, bfsstr, connectstr)

    def _check_rc(self):
        # check if pyfehmrc file exists
        rc_wd1 = os.getcwd() + os.sep + ".pylagritrc"
        rc_wd2 = os.getcwd() + os.sep + "pylagritrc"
        rc_home1 = os.path.expanduser("~") + os.sep + ".pylagritrc"
        rc_home2 = os.path.expanduser("~") + os.sep + "pylagritrc"
        if os.path.isfile(rc_wd1):
            fp = open(rc_wd1)
        elif os.path.isfile(rc_wd2):
            fp = open(rc_wd2)
        elif os.path.isfile(rc_home1):
            fp = open(rc_home1)
        elif os.path.isfile(rc_home2):
            fp = open(rc_home2)
        else:
            return
        lns = fp.readlines()
        for ln in lns:
            ln = ln.split("#")[0]  # strip off the comment
            if ln.startswith("#"):
                continue
            elif ln.strip() == "":
                continue
            elif ":" in ln:
                v = ln.split(":")
                if v[0].strip() == "lagrit_exe":
                    self.lagrit_exe = os.path.expanduser(
                        v[1].strip().replace('"', "").replace("'", "")
                    )
                elif v[0].strip() == "gmv_exe":
                    self.gmv_exe = os.path.expanduser(
                        v[1].strip().replace('"', "").replace("'", "")
                    )
                elif v[0].strip() == "paraview_exe":
                    self.paraview_exe = os.path.expanduser(
                        v[1].strip().replace('"', "").replace("'", "")
                    )
                else:
                    print("WARNING: unrecognized .pylagritrc line '" + ln.strip() + "'")
            else:
                print("WARNING: unrecognized .pylagritrc line '" + ln.strip() + "'")

    def extract_surfmesh(
        self,
        name: Optional[str] = None,
        cmo_in: "Optional[MO | str]" = None,
        stride=(1, 0, 0),
        reorder=True,
        resetpts_itp=True,
        external=False,
        append=None,
    ) -> "MO":
        if name is None:
            name = make_name("mo", self.mo.keys())

        stride = [str(v) for v in stride]
        cmd = ["extract/surfmesh", ",".join(stride), name]

        if cmo_in is not None:
            cmo = self.mo[str(cmo_in)]

            if resetpts_itp:
                cmo.resetpts_itp()

            if reorder:
                cmo.sendcmd("createpts/median")
                self.sendcmd(
                    "/".join(
                        [
                            "sort",
                            str(cmo_in),
                            "index/ascending/ikey/itetclr zmed ymed xmed",
                        ]
                    )
                )
                self.sendcmd("/".join(["reorder", str(cmo_in), "ikey"]))
                self.sendcmd("/".join(["cmo/DELATT", str(cmo_in), "xmed"]))
                self.sendcmd("/".join(["cmo/DELATT", str(cmo_in), "ymed"]))
                self.sendcmd("/".join(["cmo/DELATT", str(cmo_in), "zmed"]))
                self.sendcmd("/".join(["cmo/DELATT", str(cmo_in), "ikey"]))

            cmd.append(str(cmo_in))

        if external:
            cmd.append("external")

        if append:
            cmd.append(append)

        self.sendcmd("/".join(cmd))
        self.mo[name] = MO(name, self)

        return self.mo[name]

    def read_script(self, fname: str):
        """
        Read a LaGriT Script

        Given a script name, executes the script in LaGriT.

        :param fname: The name or path to the lagrit script.
        :type fname: str
        """

        f = open(fname)
        commands = f.readlines()
        for c in commands:
            # Remove newlines and spaces
            c = "".join(c.split())
            if len(c) != 0 and "finish" not in c:
                self.sendcmd(c)

    def read_att(
        self,
        fname: str,
        attributes: List[str],
        mesh: Optional["MO"] = None,
        operation="add",
    ):
        """
        Reads data from a file into an attribute.
        """

        if mesh is None:
            mesh = self.create()

        if isinstance(operation, (list, tuple)):
            operation = ",".join(list(map(str, operation)))

        cmd = "/".join(
            ["cmo", "readatt", mesh.name, ",".join(attributes), operation, fname]
        )
        self.sendcmd(cmd)

        return mesh

    def define(self, **kwargs):
        """
        Pass in a variable number of arguments to be defined in
        LaGriT's internal global scope.

        Note that it is generally considered bad practice in PyLaGriT
        to rely on LaGriT's variable system for parameters; however,
        there are use-cases where it is necessary: i.e., macro scripts.

        Usage:

            lg.define(MO_PTS=mo_pts.name,OUTFILE='mesh.inp',PERTURB32=1.3244)

            >> define / MO_PTS / mo1
            >> define / OUTFILE / mesh.inp
            >> define / PERTURB32 / 1.3244

        """

        for key, value in kwargs.items():
            self.sendcmd(f"define / {key} / {value}")

    def convert(self, filename: str, to_type: str):
        """
        Convert File

        For each file of the pattern, creates a new file in the to_format format.
        The new files will be inside the directory that the LaGriT object was
        instantiated. The name of each file will be the same as the original
        file with the extension changed to to_type.

        Supports conversion from avs, and gmv files.
        Supports conversion to avs, exo, and gmv files.

        :param filename: name of the file to be converted.
        :type  pattern: str

        :param to_type: New format to convert files.
        :type  to_type: str

        Example:
            >>> # To use pylagrit, import the module.
            >>> import pylagrit
            >>>
            >>> # Create your pylagrit session.
            >>> lg = pylagrit.PyLaGriT()
            >>>
            >>> # Create a mesh object and dump it to a gmv file 'test.gmv'.
            >>> mo = lg.create(name="test")
            >>> mo.createpts_brick_xyz(
            ...     (5, 5, 5),
            ...     (0, 0, 0),
            ...     (5, 5, 5),
            ... )
            >>> mo.dump("gmv", "test.gmv")
            >>>
            >>> # Convert test.gmv to exoduce and contour files.
            >>> lg.convert("test.gmv", "exo")
            >>> lg.convert("test.gmv", "avs")
        """
        # Make sure I support the new filetype.
        if to_type not in ["avs", "gmv", "exo"]:
            raise ValueError(f"Conversion to {to_type} not supported.")

        fname = Path(filename)
        # Check that I support the old filetype.
        if fname.suffix not in [".avs", ".gmv"]:
            raise ValueError(f"Conversion from {fname.suffix} not supported.")

        cmo: MO = self.read_mo(str(filename))  # type: ignore
        cmo.dump(f"{fname.stem}.{to_type}")
        cmo.delete()

    def merge(self, mesh_objs: List["MO"], name: Optional[str] = None):
        """
        Merge Mesh Objects

        Merges two or more mesh objects together and returns the combined mesh
        object.

        :param mesh_objs: An argument list of mesh objects.
        :type  mesh_objs: MO list

        Returns: MO.

        Example:
            >>> # To use pylagrit, import the module.
            >>> import pylagrit
            >>> import numpy
            >>> # Instantiate the lagrit object.
            >>> lg = pylagrit.PyLaGriT()
            >>> # Create list with mesh object as first element
            >>> dxyz = numpy.array([0.25] * 3)
            >>> mins = numpy.array([0.0] * 3)
            >>> maxs = numpy.array([1.0] * 3)
            >>> ms = [lg.createpts_dxyz(dxyz, mins, maxs, "tet", connect=True)]
            >>> # Create three new mesh objects, each one directly above the other
            >>> for i in range(3):
            >>>     ms.append(ms[-1].copy())
            >>>     ms[-1].trans(ms[-1].mins,ms[-1].mins+numpy.array([0.,0.,1.]))
            >>> # Merge list of mesh objects and clean up
            >>> mo_merge = lg.merge(ms)
            >>> for mo in ms:
            ...     mo.delete()
            >>> mo_merge.rmpoint_compress(filter_bool=True, resetpts_itp=True)
            >>> mo_merge.paraview(filename="mo_merge.inp")
        """
        if name is None:
            name = make_name("mo", self.mo.keys())
        self.mo[name] = MO(name, self)
        if len(mesh_objs) > 1:
            for mo in mesh_objs:
                cmd = "/".join(["addmesh", "merge", name, name, mo.name])
                self.sendcmd(cmd)
        else:
            raise ValueError("Must provide at least two objects to merge.")
        return self.mo[name]

    def create(
        self, elem_type="tet", name: Optional[str] = None, npoints=0, nelements=0
    ):
        """
        Create a Mesh Object

        Creates a mesh object in lagrit and an MO in the LaGriT object. Returns
        the mesh object.

        :kwarg name: Name to be given to the mesh object.
        :type  name: str

        :kwarg mesh: The type of mesh object to create.
        :type  mesh: str

        :kwarg npoints: The number of points.
        :type  npoints: int

        :kwarg nelements: The number of elements.
        :type  nelements: int

        Returns: MO
        """

        if name is None:
            name = make_name("mo", self.mo.keys())

        self.sendcmd("cmo/create/%s/%i/%i/%s" % (name, npoints, nelements, elem_type))
        self.mo[name] = MO(name, self)
        return self.mo[name]

    def create_tet(self, name: Optional[str] = None, npoints=0, nelements=0):
        """Create a tetrahedron mesh object."""
        return self.create(elem_type="tet", **minus_self(locals()))

    def create_hex(self, name: Optional[str] = None, npoints=0, nelements=0):
        """Create a hexagon mesh object."""
        return self.create(elem_type="hex", **minus_self(locals()))

    def create_pri(self, name: Optional[str] = None, npoints=0, nelements=0):
        """Create a prism mesh object."""
        return self.create(elem_type="pri", **minus_self(locals()))

    def create_pyr(self, name: Optional[str] = None, npoints=0, nelements=0):
        """Create a pyramid mesh object."""
        return self.create(elem_type="pyr", **minus_self(locals()))

    def create_tri(self, name: Optional[str] = None, npoints=0, nelements=0):
        """Create a triangle mesh object."""
        return self.create(elem_type="tri", **minus_self(locals()))

    def create_qua(self, name: Optional[str] = None, npoints=0, nelements=0):
        """Create a quadrilateral mesh object."""
        return self.create(elem_type="qua", **minus_self(locals()))

    def create_hyb(self, name: Optional[str] = None, npoints=0, nelements=0):
        """Create a hybrid mesh object."""
        return self.create(elem_type="hyb", **minus_self(locals()))

    def create_line(
        self,
        npoints=0,
        mins: Optional[Tuple[float, float, float]] = None,
        maxs: Optional[Tuple[float, float, float]] = None,
        rz_switch=(1, 1, 1),
        name: Optional[str] = None,
    ):
        """Create a line mesh object."""
        mo_new = self.create(elem_type="lin", name=name, npoints=npoints)
        if mins is not None and maxs is not None:
            mo_new.createpts_line(npoints, mins, maxs, rz_switch)
        return mo_new

    def create_triplane(self, name: Optional[str] = None, npoints=0, nelements=0):
        """Create a triplane mesh object."""
        return self.create(elem_type="triplane", **minus_self(locals()))

    def copy(self, mo: "MO | str", name: Optional[str] = None):
        """
        Copy Mesh Object

        Copies a mesh object, mo, and returns the MO object.
        """

        # Check if name was specified, if not just generate one.
        if name is None:
            name = make_name("mo", self.mo.keys())

        # Create the MO in lagrit and the PyLaGriT object.
        self.sendcmd(f"cmo/copy/{name}/{str(mo)}")
        self.mo[name] = MO(name, self)

        return self.mo[name]

    def dump(self, filename: str, mos: List["MO"] = [], filetype="binary"):  # noqa: B006
        """
        Dump lagrit binary file
        :arg filename: name of lagrit binary file to create
        :type filename: string
        :arg mos: List of mesh objects to include, default is all
        :type mos: list(MO)
        :arg filetype: Filetype to dump, 'binary' or 'ascii'
        :type mos: string
        """
        cmd = ["dump", "lagrit", filename]
        if len(mos) == 0:
            cmd.append("-all-")
        else:
            cmd += ",".join([mo.name for mo in mos])
        if filetype == "ascii":
            cmd.append("ascii")
        self.sendcmd("/".join(cmd))

    def tri_mo_from_polyline(
        self,
        coords: List[Tuple[float, float]],
        filename="polyline.inp",
        name: Optional[str] = None,
    ):
        """
        Create polygon tri mesh object from points
        Points are expected to be defined clockwise by default

        :param coords: x,y,z coordinates defined in npoints by 3 array, points expected to be ordered clockwise by default
        :type coords: lst(floats) or ndarray(floats)
        :param order: ordering of points, clockwise by default
        :type order: string
        :param filename: Name of avs polyline file to create
        :type filename: string
        :param name: Internal lagrit name for mesh object
        :type name: string
        :returns: PyLaGriT Mesh Object

        Example:
            >>> from pylagrit import PyLaGriT
            >>> lg = PyLaGriT()
            >>> mo = lg.tri_mo_from_polyline(
            ...     [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
            ... )
        """
        mstr = str(len(coords)) + " " + str(len(coords)) + " 0 0 0\n"
        for i, p in enumerate(coords):
            mstr += " ".join([str(i + 1), str(p[0]), str(p[1]), str(0.0)])
            mstr += "\n"
        es1 = numpy.arange(len(coords)) + 1
        es2 = numpy.roll(es1, len(coords) - 1)
        for e1, e2 in zip(es1, es2):
            mstr += " ".join([str(e1), "1 line ", str(e1), str(e2)])
            mstr += "\n"
        with open(filename, "w") as fh:
            fh.write(mstr)
        # Check if name was specified, if not just generate one.
        if name is None:
            name = make_name("mo", self.mo.keys())
        motmp = cast(MO, self.read_mo(filename))
        motri = motmp.copypts(elem_type="tri")
        motmp.delete()
        self.mo[name] = motri
        return self.mo[name]

    def createpts(
        self,
        crd: str,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        elem_type: str,
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_value=(1, 1, 1),
        connect=False,
        name: Optional[str] = None,
    ):
        """
        Create and Connect Points

        :arg crd: Coordinate type of either 'xyz' (cartesian coordinates),
                    'rtz' (cylindrical coordinates), or
                    'rtp' (spherical coordinates).
        :type  crd: str
        :arg  npts: The number of points to create in line
        :type npts: tuple(int)
        :arg  mins: The starting value for each dimension.
        :type mins: tuple(int, int, int)
        :arg  maxs: The ending value for each dimension.
        :type maxs: tuple(int, int, int)
        :kwarg elem_type: The type of mesh object to create
        :type  elem_type: str
        :kwarg vc_switch: Determines if nodes represent vertices (1) or cell centers (0).
        :type  vc_switch: tuple(int, int, int)
        :kwarg rz_switch: Determines true or false (1 or 0) for using ratio zoning values.
        :type  rz_switch: tuple(int, int, int)
        :returns: MO

        """
        if elem_type.startswith(("triplane", "qua")):
            assert numpy.where(numpy.array(npts) <= 1)[0].shape[0] == 1, (  # noqa: S101
                f"{elem_type} elem_type requires one (1) in npts"
            )
            assert (  # noqa: S101
                numpy.where((numpy.array(maxs) - numpy.array(mins)) == 0)[0][0] == 1
            ), f"{elem_type} elem_type requires one zero range (max-min)"
        if elem_type.startswith(("tet", "pri", "pyr", "hex")):
            assert numpy.all(numpy.array(npts) > 1), (  # noqa: S101
                f"{elem_type} elem_type requires all npts greater than 1"
            )
            assert numpy.all((numpy.array(maxs) - numpy.array(mins)) > 0), (  # noqa: S101
                f"{elem_type} elem_type requires all ranges (max-min) greater than 0"
            )
        mo = self.create(elem_type=elem_type, name=name)
        mo.createpts(
            crd,
            npts,
            mins,
            maxs,
            vc_switch=vc_switch,
            rz_switch=rz_switch,
            rz_value=rz_value,
            connect=connect,
        )
        return mo

    def createpts_xyz(
        self,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        elem_type: str,
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_value=(1, 1, 1),
        connect=True,
        name: Optional[str] = None,
    ):
        return self.createpts(
            "xyz",
            npts,
            mins,
            maxs,
            elem_type,
            vc_switch,
            rz_switch,
            rz_value,
            connect=connect,
            name=name,
        )

    def createpts_dxyz(
        self,
        dxyz: Tuple[float, float, float],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        elem_type: str,
        clip="under",
        hard_bound: str | Tuple[str, str, str] = "min",
        rz_switch=(1, 1, 1),
        rz_value=(1, 1, 1),
        connect=True,
        name: Optional[str] = None,
    ):
        """
        Create and Connect Points to create an orthogonal hexahedral mesh. The
        vertex spacing is based on dxyz and the mins and maxs specified. mins
        (default, see hard_bound option) or maxs will be adhered to, while maxs
        (default) or mins will be modified based on the clip option to be
        truncated at the nearest value 'under' (default) or 'over' the range
        maxs-mins. clip and hard_bound options can be mixed by specifying tuples
        (see description below).

        :arg  dxyz: The spacing between points in x, y, and z directions
        :type dxyz: tuple(float,float,float)
        :arg  mins: The starting value for each dimension.
        :type mins: tuple(float,float,float)
        :arg  maxs: The ending value for each dimension.
        :type maxs: tuple(float,float,float)
        :kwarg mesh: The type of mesh object to create, automatically set to 'triplane' if 2d or 'tet' if 3d.
        :type  mesh: str
        :kwarg clip: How to handle bounds if range does not divide by dxyz, either clip 'under' or 'over' range
        :type clip: string or tuple(string,string,string)
        :kwarg hard_bound: Whether to use the "min" or "max" as the hard constraint on dimension
        :type hard_bound: string or tuple(string,string,string)
        :kwarg rz_switch: Determines true or false (1 or 0) for using ratio zoning values.
        :type  rz_switch: tuple(int, int, int)
        :kwarg connect: Whether or not to connect points
        :type  connect: boolean

        Example:
            >>> from pylagrit import PyLaGriT
            >>> l = PyLaGriT()
            >>>
            >>> # Create 2x2x2 cell mesh
            >>> m = l.create()
            >>> m.createpts_dxyz(
            ...     (0.5, 0.5, 0.5),
            ...     (0.0, 0.0, 0.0),
            ...     (1.0, 1.0, 1.0),
            ...     rz_switch=[1, 1, 1],
            ...     connect=True,
            ... )
            >>> m.paraview()
            >>> # m.gmv()
            >>>
            >>> # Create 2x2x2 mesh where maxs will be truncated to nearest value under given maxs
            >>> m_under = l.create()
            >>> m_under.createpts_dxyz(
            ...     (0.4, 0.4, 0.4),
            ...     (0.0, 0.0, 0.0),
            ...     (1.0, 1.0, 1.0),
            ...     rz_switch=[1, 1, 1],
            ...     connect=True,
            ... )
            >>> m_under.paraview()
            >>> # m_under.gmv()
            >>>
            >>> # Create 3x3x3 mesh where maxs will be truncated to nearest value over given maxs
            >>> m_over = l.create()
            >>> m_over.createpts_dxyz(
            ...     (0.4, 0.4, 0.4),
            ...     (0.0, 0.0, 0.0),
            ...     (1.0, 1.0, 1.0),
            ...     clip="over",
            ...     rz_switch=[1, 1, 1],
            ...     connect=True,
            ... )
            >>> m_over.paraview()
            >>> # m_over.gmv()
            >>>
            >>> # Create 3x3x3 mesh where x and y maxs will be truncated to nearest value over given maxs
            >>> # and z min will be truncated  to nearest value
            >>> m_mixed = l.create()
            >>> m_mixed.createpts_dxyz(
            ...     (0.4, 0.4, 0.4),
            ...     (0.0, 0.0, -1.0),
            ...     (1.0, 1.0, 0.0),
            ...     hard_bound=("min", "min", "max"),
            ...     clip=("under", "under", "over"),
            ...     rz_switch=[1, 1, 1],
            ...     connect=True,
            ... )
            >>> m_mixed.paraview()
            >>> # m_mixed.gmv()
        """
        mo = self.create(elem_type=elem_type, name=name)
        mo.createpts_dxyz(
            dxyz,
            mins,
            maxs,
            clip=clip,
            hard_bound=hard_bound,
            rz_switch=rz_switch,
            rz_value=rz_value,
            connect=connect,
        )
        return mo

    def createpts_rtz(
        self,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        elem_type: str,
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_value=(1, 1, 1),
        connect=True,
    ):
        return self.createpts(
            "rtz",
            npts,
            mins,
            maxs,
            elem_type,
            vc_switch,
            rz_switch,
            rz_value,
            connect=connect,
        )

    def createpts_rtp(
        self,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        elem_type: str,
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_value=(1, 1, 1),
        connect=True,
    ):
        return self.createpts(
            "rtp",
            npts,
            mins,
            maxs,
            elem_type,
            vc_switch,
            rz_switch,
            rz_value,
            connect=connect,
        )

    def createpts_line(
        self,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        elem_type="line",
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        name: Optional[str] = None,
    ):
        """
        Create and Connect Points in a line

        :arg  npts: The number of points to create in line
        :type npts: int
        :arg  mins: The starting value for each dimension.
        :type mins: tuple(int, int, int)
        :arg  maxs: The ending value for each dimension.
        :type maxs: tuple(int, int, int)
        :kwarg vc_switch: Determines if nodes represent vertices (1) or cell centers (0).
        :type  vc_switch: tuple(int, int, int)
        :kwarg rz_switch: Determines true or false (1 or 0) for using ratio zoning values.
        :type  rz_switch: tuple(int, int, int)

        """
        mo = self.create(elem_type, name=name)
        mo.createpts_line(npts, mins, maxs, vc_switch=vc_switch, rz_switch=rz_switch)
        return mo

    def gridder(
        self,
        x: Optional[List[float]] = None,
        y: Optional[List[float]] = None,
        z: Optional[List[float]] = None,
        connect=False,
        elem_type="tet",
        name: Optional[str] = None,
        filename="gridder.inp",
    ):
        """
        Generate a logically rectangular orthogonal mesh corresponding to vectors of nodal positions.

        :arg x: x discretization locations
        :type x: array(floats)
        :arg y: y discretization locations
        :type y: array(floats)
        :arg z: z discretization locations
        :type z: array(floats)
        :arg connect: Should the points be connected
        :type connect: bool
        :arg elem_type: Type of element for created mesh object
        :type elem_type: string
        :arg filename: Name of avs file created with nodal coordinates
        :type filename: string
        :returns: MO

        Example:
            >>> from pylagrit import PyLaGriT
            >>> import numpy
            >>> lg = PyLaGriT()
            >>> x0 = -numpy.logspace(1, 2, 15, endpoint=True)
            >>> x1 = numpy.arange(-10, 10, 1)
            >>> x2 = -x0
            >>> x = numpy.concatenate([x0, x1, x2])
            >>> y = x
            >>> mqua = lg.gridder(x, y, elem_type="quad", connect=True)
            >>> mqua.paraview()
        """
        # TODO: validation for point set
        # dim = 0
        # if x is not None:
        #     if len(x) > 0:
        #         dim += 1
        # if y is not None:
        #     if len(y) > 0:
        #         dim += 1
        # if z is not None:
        #     if len(z) > 0:
        #         dim += 1
        # if dim == 0:
        #     print("ERROR: must define at least one of x, y, z arrays")
        #     return
        # if elem_type in ["line"] and dim != 1:
        #     print(
        #         "Error: Only 1 coordinate array (x,y,z) required for elem_type 'line'"
        #     )
        #     return
        # if elem_type in ["tri", "quad"] and dim != 2:
        #     print(
        #         "Error: Only 2 coordinate arrays (x,y,z) required for elem_type '"
        #         + str(elem_type)
        #         + "'"
        #     )
        #     return
        # if elem_type in ["tet", "hex"] and dim != 3:
        #     print(
        #         "Error: 3 coordinate arrays (x,y,z) required for elem_type '"
        #         + str(elem_type)
        #         + "'"
        #     )
        #     print("Set elem_type to a 2D format like 'quad' or 'triplane'")
        #     return
        if x is None or len(x) == 0:
            x = [0]
        if y is None or len(y) == 0:
            y = [0]
        if z is None or len(z) == 0:
            z = [0]
        x = list(numpy.unique(x))
        y = list(numpy.unique(y))
        z = list(numpy.unique(z))
        nodelist = numpy.array(list(product(*[z, y, x])))
        nodelist = numpy.fliplr(nodelist)

        outfile = open(filename, "w")
        outfile.write(f"   {len(nodelist)} 0 0 0 0\n")
        for i, nd in enumerate(nodelist):
            outfile.write(f"{i:11d}        ")
            outfile.write(f"{nd[0]:14.8f}        ")
            outfile.write(f"{nd[1]:14.8f}        ")
            outfile.write(f"{nd[2]:14.8f}")
            outfile.write("\n")
        outfile.write("\n")
        outfile.close()

        m = (
            self.create(elem_type)
            if name is None
            else self.create(elem_type, name=name)
        )
        m.read(filename)

        if elem_type in ["quad", "hex"] and connect:
            cmd = [
                "createpts",
                "brick",
                "xyz",
                " ".join([str(len(x)), str(len(y)), str(len(z))]),
                "1 0 0",
                "connect",
            ]
            m.sendcmd("/".join(cmd))
        elif connect:
            m.connect()

        self.sendcmd(f"cmo/printatt/{m.name}/-xyz- minmax")
        return m

    def points(
        self,
        coords: List[Tuple[float, float, float]],
        connect=False,
        elem_type="tet",
        filename="points.inp",
    ):
        """
        Generate a mesh object of points defined by x, y, z vectors.

        :arg coords: list of 3-tuples containing (x,y,z) coorinates
        :type x: array(3-tuples), npoints by 3 array
        :arg connect: Should the points be connected
        :type connect: bool
        :arg elem_type: Type of element for created mesh object
        :type elem_type: string
        :arg filename: Name of avs file created with nodal coordinates
        :type filename: string
        :returns: MO

        Example:
            >>> from pylagrit import PyLaGriT
            >>> lg = PyLaGriT()
            >>> coords = [
            ...     [0, 0, 0],
            ...     [1, 0, 0],
            ...     [1, 1, 0],
            ...     [0, 1, 1],
            ...     [0, 0, 1],
            ...     [0, 1, 0],
            ...     [1, 1, 1],
            ...     [1, 0, 1],
            ... ]
            >>> m = lg.points(coords, elem_type="tet", connect=True)
            >>> m.paraview()
        """
        # TODO: validation for point set
        # dim = 0
        # ix = numpy.all(numpy.diff(coords[:, 0]) == 0)
        # if not ix:
        #     dim += 1
        # iy = numpy.all(numpy.diff(coords[:, 1]) == 0)
        # if not iy:
        #     dim += 1
        # iz = numpy.all(numpy.diff(coords[:, 2]) == 0)
        # if not iz:
        #     dim += 1
        # if elem_type in ["line"] and dim != 1:
        #     print("Error: Coordinates must form line for elem_type 'line'")
        #     return
        # if elem_type in ["tri", "quad"] and dim != 2:
        #     print(
        #         "Error: Coordinates must form plane for elem_type '"
        #         + str(elem_type)
        #         + "'"
        #     )
        #     return
        # if elem_type in ["tet", "hex"] and dim != 3:
        #     print(
        #         "Error: 3D coordinates required for elem_type '" + str(elem_type) + "'"
        #     )
        #     print("Set elem_type to a 2D format like 'quad' or 'triplane'")
        #     return

        outfile = open(filename, "w")
        outfile.write("   " + str(len(coords)) + " 0 0 0 0\n")
        for i, nd in enumerate(coords):
            outfile.write(f"{i:11d}        ")
            outfile.write(f"{nd[0]:14.8f}        ")
            outfile.write(f"{nd[1]:14.8f}        ")
            outfile.write(f"{nd[2]:14.8f}")
            outfile.write("\n")
        outfile.write("\n")
        outfile.close()
        m = self.create(elem_type)
        m.read(filename)
        if elem_type in ["quad", "hex"] and connect:
            cmd = [
                "createpts",
                "brick",
                "xyz",
                " ".join([str(len(coords)), str(len(coords)), str(len(coords))]),
                "1 0 0",
                "connect",
            ]
            m.sendcmd("/".join(cmd))
        elif connect:
            m.connect()
        return m


class MO:
    """Mesh object class"""

    def __init__(self, name: str, parent: PyLaGriT):
        self.name = name
        self._parent = parent
        self.pset: Dict[str, PSet] = {}
        self.eltset: Dict[str, EltSet] = {}
        self.regions: Dict[str, Region] = {}
        self.mregions: Dict[str, MRegion] = {}
        self.surfaces: Dict[str, Surface] = {}

    def __repr__(self):
        return self.name

    def sendcmd(self, cmd: str, verbose=True, expectstr="Enter a command"):
        self._parent.sendcmd(
            "cmo select " + self.name, verbose=verbose, expectstr=expectstr
        )
        self._parent.sendcmd(cmd, verbose=verbose, expectstr=expectstr)

    @property
    def mins(self):
        return numpy.array([self.xmin, self.ymin, self.zmin])

    @property
    def maxs(self):
        return numpy.array([self.xmax, self.ymax, self.zmax])

    @property
    def xmin(self):
        self.minmax_xyz(verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return float(strarr[4].split()[1])

    @property
    def xmax(self):
        self.minmax_xyz(verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return float(strarr[4].split()[2])

    @property
    def xlength(self):
        self.minmax_xyz(verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return int(strarr[4].split()[4])

    @property
    def ymin(self):
        self.minmax_xyz(verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return float(strarr[5].split()[1])

    @property
    def ymax(self):
        self.minmax_xyz(verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return float(strarr[5].split()[2])

    @property
    def ylength(self):
        self.minmax_xyz(verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return int(strarr[5].split()[4])

    @property
    def zmin(self):
        self.minmax_xyz(verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return float(strarr[6].split()[1])

    @property
    def zmax(self):
        self.minmax_xyz(verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return float(strarr[6].split()[2])

    @property
    def zlength(self):
        self.minmax_xyz(verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return int(strarr[6].split()[4])

    @property
    def nnodes(self):
        self.status(brief=True, verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return int(strarr[7].split()[4])

    @property
    def nelems(self):
        self.status(brief=True, verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return int(strarr[7].split()[-1])

    @property
    def ndim_geo(self):
        self.status(brief=True, verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return int(strarr[8].split()[3])

    @property
    def ndim_topo(self):
        self.status(brief=True, verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        return int(strarr[9].split()[3])

    @property
    def elem_type(self):
        self.status(brief=True, verbose=False)
        strarr = cast(bytes, self._parent.before).splitlines()
        etype = decode_binary(strarr[8].split()[7])
        if etype == "tri":
            if self.ndim_geo == 2:
                etype = "triplane"
        return etype

    def status(self, brief=False, verbose=True):
        print(self.name)
        self._parent.cmo_status(self.name, brief=brief, verbose=verbose)

    def select(self):
        self.sendcmd("cmo/select/" + self.name)

    def read(self, filename: str, filetype: Optional[str] = None):
        # If filetype is lagrit, name is irrelevant
        if filetype is not None:
            cmd = "/".join(["read", filetype])
        else:
            cmd = "read"
        if filetype != "lagrit":
            cmd = "/".join([cmd, filename, self.name])
        else:
            print("Error: Can't read in lagrit type file into existing mesh object")
            return
        self.sendcmd(cmd)

    def printatt(
        self,
        attname: Optional[str] = None,
        stride=(1, 0, 0),
        pset: Optional["PSet"] = None,
        eltset: Optional["EltSet"] = None,
        ptype="value",
    ):
        stride = [str(v) for v in stride]

        if attname is None:
            attname = "-all-"

        if pset is not None:
            cmd = "/".join(
                [
                    "cmo/printatt",
                    self.name,
                    attname,
                    ptype,
                    ",".join(["pset", "get", str(pset)]),
                ]
            )
        elif eltset is not None:
            cmd = "/".join(
                [
                    "cmo/printatt",
                    self.name,
                    attname,
                    ptype,
                    ",".join(["eltset", "get", str(eltset)]),
                ]
            )
        else:
            cmd = "/".join(
                ["cmo/printatt", self.name, attname, ptype, ",".join(stride)]
            )

        self.sendcmd(cmd)

    def delatt(self, attnames: List[str], force=True):
        """
        Delete a list of attributes

        :arg attnames: Attribute names to delete
        :type attnames: str or lst(str)
        :arg force: If true, delete even if the attribute permanent persistance
        :type force: bool

        """
        # If single attribute as string, make list
        if isinstance(attnames, str):
            attnames = [attnames]
        for att in attnames:
            if force:
                cmd = "/".join(["cmo/DELATT", self.name, att])
            else:
                cmd = "/".join(["cmo/delatt", self.name, att])
            self.sendcmd(cmd)

    def copyatt(
        self,
        attname_src: str,
        attname_sink: Optional[str] = None,
        mo_src: Optional["MO"] = None,
    ):
        """
        Add a list of attributes

        :arg attname_src: Name of attribute to copy
        :type attname_src: str
        :arg attname_sink: Name of sink attribute
        :type attname_sink: str
        :arg mo_src: Name of source mesh object
        :type mo_src: PyLaGriT Mesh Object

        """
        if attname_sink is None:
            attname_sink = attname_src
        if mo_src is None:
            mo_src = self
        cmd = "/".join(
            ["cmo/copyatt", self.name, mo_src.name, attname_sink, attname_src]
        )
        self.sendcmd(cmd)

    def add_element_attribute(
        self,
        attname: str,
        keyword: Optional[str] = None,
        vtype="VDOUBLE",
        rank="scalar",
        interpolate="linear",
        persistence="permanent",
        ioflag="",
        value=0.0,
    ):
        """
        Add a list of attributes to elements

        :arg attnames: Attribute name to add
        :type attnames: str
        :arg keyword: Keyword used by lagrit for specific attributes
        :type name: str
        :arg vtype: Type of variable {'VDOUBLE','VINT',...}
        :type name: str

        """
        self.addatt(
            attname,
            keyword=keyword,
            vtype=vtype,
            rank=rank,
            length="nelements",
            interpolate=interpolate,
            persistence=persistence,
            ioflag=ioflag,
            value=value,
        )

    def add_node_attribute(
        self,
        attname: str,
        keyword: Optional[str] = None,
        vtype="VDOUBLE",
        rank="scalar",
        interpolate="linear",
        persistence="permanent",
        ioflag="",
        value=0.0,
    ):
        """
        Add a list of attributes to nodes

        :arg attnames: Attribute name to add
        :type attnames: str
        :arg keyword: Keyword used by lagrit for specific attributes
        :type name: str
        :arg vtype: Type of variable {'VDOUBLE','VINT',...}
        :type name: str

        """
        self.addatt(
            attname,
            keyword=keyword,
            vtype=vtype,
            rank=rank,
            length="nnodes",
            interpolate=interpolate,
            persistence=persistence,
            ioflag=ioflag,
            value=value,
        )

    def addatt(
        self,
        attname: str,
        keyword: Optional[str] = None,
        vtype="VDOUBLE",
        rank="scalar",
        length="nnodes",
        interpolate="linear",
        persistence="permanent",
        ioflag="",
        value=0.0,
    ):
        """
        Add a list of attributes

        :arg attnames: Attribute name to add
        :type attnames: str
        :arg keyword: Keyword used by lagrit for specific attributes
        :type name: str
        :arg vtype: Type of variable {'VDOUBLE','VINT',...}
        :type name: str

        """
        if keyword is not None:
            cmd = "/".join(["cmo/addatt", self.name, keyword, attname])
        else:
            cmd = "/".join(
                [
                    "cmo/addatt",
                    self.name,
                    attname,
                    vtype,
                    rank,
                    length,
                    interpolate,
                    persistence,
                    ioflag,
                    str(value),
                ]
            )
        self.sendcmd(cmd)

    def addatt_voronoi_volume(self, name="voronoi_volume"):
        """
        Add voronoi volume attribute to mesh object

        :arg name: name of attribute in LaGriT
        :type name: str
        """
        self.addatt(name, keyword="voronoi_volume")

    def addatt_voronoi_varea(self, attr_names="xvarea yvarea zvarea"):
        """
        Add voronoi area x,y,z component attributes for 2D planar mesh

        :arg attr_names: name of x,y,z attributes in LaGriT
        :type name: str
        """
        self.addatt(attr_names, keyword="voronoi_varea")

    def minmax(self, attname: Optional[str] = None, stride=(1, 0, 0)):
        self.printatt(attname=attname, stride=stride, ptype="minmax")

    def minmax_xyz(self, stride=(1, 0, 0), verbose=True):
        cmd = "/".join(["cmo/printatt", self.name, "-xyz-", "minmax"])
        self.sendcmd(cmd, verbose=verbose)

    def list(
        self,
        attname: Optional[str] = None,
        stride=(1, 0, 0),
        pset: Optional["PSet"] = None,
    ):
        self.printatt(attname=attname, stride=stride, pset=pset, ptype="list")

    def setatt(self, attname: str, value: int | float, stride=(1, 0, 0)):
        stride = [str(v) for v in stride]
        cmd = "/".join(["cmo/setatt", self.name, attname, ",".join(stride), str(value)])
        self.sendcmd(cmd)

    def set_id(self, option: str, node_attname="id_node", elem_attname="id_elem"):
        """
        This command creates integer attributes that contain the node and/or
        element number. If later operations delete nodes or
        elements causing renumbering, these attributes will contain the
        original node or element number.

        :arg option: create attribute for nodes, elements, or both {'both','node','element'}
        :type option: str

        :arg node_attname: name for new node attribute
        :type node_attname: str

        :arg elem_attname: name for new element attribute
        :type elem_attname: str

        Example:
        from pylagrit import PyLaGriT
        #instantiate PyLaGriT
        lg = PyLaGriT()
        #create source mesh
        npts = (11,11,11)
        mins = (0.,0.,0.)
        maxs = (1.,1.,1.)
        mesh = lg.create()
        mesh.createpts_brick_xyz(npts,mins,maxs)
        #write node and element attribute numbers
        mesh.set_id('both',node_attname='node_att1',elem_attname='elem_att1')
        #select and remove points
        p_mins = (0.5,0.,0.)
        p_maxs = (1.,1.,1.)
        points = mesh.pset_geom_xyz(p_mins,p_maxs)
        mesh.rmpoint_pset(points)
        #dump mesh with original node and element numbering saved
        mesh.dump('set_id_test.gmv')
        """
        if option == "both":
            cmd = "/".join(
                ["cmo/set_id", self.name, option, node_attname, elem_attname]
            )
        elif option == "node":
            cmd = "/".join(["cmo/set_id", self.name, option, node_attname])
        elif option == "element":
            cmd = "/".join(["cmo/set_id", self.name, option, elem_attname])
        else:
            print("ERROR: 'option' must be 'both' or 'node' or 'element'")
            return
        self.sendcmd(cmd)

    def information(self):
        """
        Returns a formatted dictionary with mesh information.

        Information is that found in cmo/status/MO
        """
        import contextlib

        @contextlib.contextmanager
        def capture():
            import sys

            from io import StringIO

            oldout, olderr = sys.stdout, sys.stderr
            try:
                out = [StringIO(), StringIO()]
                sys.stdout, sys.stderr = out
                yield out
            finally:
                sys.stdout, sys.stderr = oldout, olderr
                # FIXME: out is possibly unbound
                out[0] = out[0].getvalue()  # type: ignore
                out[1] = out[1].getvalue()  # type: ignore

        _temp = self._parent.verbose
        self._parent.verbose = True
        with capture() as out:
            self.sendcmd("cmo/status/" + self.name, verbose=True)
        self._parent.verbose = _temp

        atts = {}
        in_attributes_section = False

        for line in cast(str, out[0]).replace("\r", "").split("\n"):
            lline = line.strip().lower()
            split = line.strip().split()

            if not in_attributes_section:
                if "number of nodes" in lline:
                    atts["nodes"] = int(split[4])
                if "number of elements" in lline:
                    atts["elements"] = int(split[-1])
                if "dimensions geometry" in lline:
                    atts["dimensions"] = int(split[3])
                if "element type" in lline:
                    atts["type"] = split[-1]
                if "dimensions topology" in lline:
                    atts["dimensions_topology"] = int(split[3])
                if "name" and "type" and "rank" and "length" in lline:
                    in_attributes_section = True
                    atts["attributes"] = {}

            else:
                try:
                    name, atype, rank, length, inter, persi, io, value = split[1:]
                except ValueError:
                    continue

                atts["attributes"][name] = {}
                atts["attributes"][name]["type"] = atype
                atts["attributes"][name]["rank"] = rank
                atts["attributes"][name]["length"] = length
                atts["attributes"][name]["inter"] = inter
                atts["attributes"][name]["persi"] = persi
                atts["attributes"][name]["io"] = io

                try:
                    atts["attributes"][name]["value"] = float(value)
                except ValueError:
                    atts["attributes"][name]["value"] = value

        return atts

    def pset_geom(
        self,
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        ctr=(0, 0, 0),
        geom="xyz",
        stride=(1, 0, 0),
        name: Optional[str] = None,
    ):
        """
        Define PSet by Geometry

        Selects points from geometry specified by string geom and returns a
        PSet.

        :arg  mins: Coordinate of one of the shape's defining points.
                     xyz (Cartesian):   (x1, y1, z1);
                     rtz (Cylindrical): (radius1, theta1, z1);
                     rtp (Spherical):   (radius1, theta1, phi1);
        :type mins: tuple(int, int, int)

        :arg  maxs: Coordinate of one of the shape's defining points.
                     xyz (Cartesian):   (x2, y2, z2);
                     rtz (Cylindrical): (radius2, theta2, z2);
                     rtp (Spherical):   (radius2, theta2, phi2);
        :type maxs: tuple(int, int, int)

        :kwarg ctr: Coordinate of the relative center.
        :type  ctr: tuple(int, int, int)

        :kwarg geom: Type of geometric shape: 'xyz' (spherical),
                     'rtz' (cylindrical), 'rtp' (spherical)
        :type  geom: str

        :kwarg stride: Nodes defined by ifirst, ilast, and istride.
        :type  stride: list[int, int, int]

        :kwarg name: The name to be assigned to the PSet created.
        :type  name: str

        Returns: PSet object
        """

        if name is None:
            name = make_name("p", self.pset.keys())

        cmd = "/".join(
            [
                "pset",
                name,
                "geom",
                geom,
                ",".join([str(v) for v in stride]),
                ",".join([str(v) for v in mins]),
                ",".join([str(v) for v in maxs]),
                ",".join([str(v) for v in ctr]),
            ]
        )
        self.sendcmd(cmd)
        self.pset[name] = PSet(name, self)

        return self.pset[name]

    def pset_geom_xyz(
        self,
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        ctr=(0, 0, 0),
        stride=(1, 0, 0),
        name: Optional[str] = None,
    ):
        """
        Define PSet by Tetrahedral Geometry

        Selects points from a Tetrahedral region.

        :arg  mins: Coordinate point of 1 of the tetrahedral's corners.
        :type mins: tuple(int, int, int)

        :arg  maxs: Coordinate point of 1 of the tetrahedral's corners.
        :type maxs: tuple(int, int, int)

        :kwarg ctr: Coordinate of the relative center.
        :type  ctr: tuple(int, int, int)

        :kwarg stride: Nodes defined by ifirst, ilast, and istride.
        :type  stride: list[int, int, int]

        :kwarg name: The name to be assigned to the PSet created.
        :type  name: str

        Returns: PSet object
        """
        return self.pset_geom(geom="xyz", **minus_self(locals()))

    def pset_geom_rtz(
        self,
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        ctr=(0, 0, 0),
        stride=(1, 0, 0),
        name: Optional[str] = None,
    ):
        """
        Forms a pset of nodes within the cylinder or cylindrical shell section
        given by radius1 to radius2, and angles theta1 to theta2 and height z1 to z2.
        Refer to http://lagrit.lanl.gov/docs/conventions.html for an explanation of angles

        :arg  mins: Defines radius1, theta1, and z1.
        :type mins: tuple(int, int, int)

        :arg  maxs: Defines radius2, theta2, and z2.
        :type maxs: tuple(int, int, int)

        :kwarg stride: Nodes defined by ifirst, ilast, and istride.
        :type  stride: list[int, int, int]

        :kwarg name: The name to be assigned to the PSet created.
        :type  name: str

        :kwarg ctr: Coordinate of the relative center.
        :type  ctr: tuple(int, int, int)

        :kwarg stride: Nodes defined by ifirst, ilast, and istride.
        :type  stride: list[int, int, int]

        :kwarg name: The name to be assigned to the PSet created.
        :type  name: str

        Returns: PSet object
        """
        return self.pset_geom(geom="rtz", **minus_self(locals()))

    def pset_geom_rtp(
        self,
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        ctr=(0, 0, 0),
        stride=(1, 0, 0),
        name: Optional[str] = None,
    ):
        """
        Forms a pset of nodes within the sphere, sperical shell or sperical section
        given by radius1 to radius2, and angles theta1 to theta2 (0 - 180) and angles
        phi1 to phi2 (0 - 360).
        Refer to http://lagrit.lanl.gov/docs/conventions.html for an explanation of angles

        :arg  mins: Defines radius1, theta1, and phi1.
        :type mins: tuple(int, int, int)

        :arg  maxs: Defines radius2, theta2, and phi2.
        :type maxs: tuple(int, int, int)

        :kwarg stride: Nodes defined by ifirst, ilast, and istride.
        :type  stride: list[int, int, int]

        :kwarg name: The name to be assigned to the PSet created.
        :type  name: str

        :kwarg ctr: Coordinate of the relative center.
        :type  ctr: tuple(int, int, int)

        :kwarg stride: Nodes defined by ifirst, ilast, and istride.
        :type  stride: list[int, int, int]

        :kwarg name: The name to be assigned to the PSet created.
        :type  name: str

        Returns: PSet object
        """
        return self.pset_geom(geom="rtp", **minus_self(locals()))

    def pset_attribute(
        self,
        attribute: str,
        value: int | float,
        comparison="eq",
        stride=(1, 0, 0),
        name: Optional[str] = None,
    ):
        """
        Define PSet by attribute

        :kwarg attribute: Nodes defined by attribute ID.
        :type  attribute: str

        :kwarg value: attribute ID value.
        :type  value: integer

        :kwarg comparison: attribute comparison, default is eq.
        :type  comparison: can use default without specifiy anything, or list[lt|le|gt|ge|eq|ne]

        :kwarg stride: Nodes defined by ifirst, ilast, and istride.
        :type  stride: list[int, int, int]

        :kwarg name: The name to be assigned to the PSet created.
        :type  name: str

        Returns: PSet object
        """
        if name is None:
            name = make_name("p", self.pset.keys())

        stride = [str(v) for v in stride]

        cmd = "/".join(
            [
                "pset",
                name,
                "attribute",
                attribute,
                ",".join(stride),
                str(value),
                comparison,
            ]
        )
        self.sendcmd(cmd)
        self.pset[name] = PSet(name, self)

        return self.pset[name]

    def compute_distance(self, mo: "MO", option="distance_field", attname="dfield"):
        """
        Compute distance from one mesh object to another

        :kwarg mo: Mesh object to compute distance to base mesh from
        :type  mo: LaGriT mesh object

        :kwarg option: The type of distance field calculation. Available choices
         are 'distance_field' and 'signed_distance_field'.
        :type  option: str

        :kwarg attname: The name of the attribute to be created in the base mesh.
        :type  attname: str

        Returns: New attribute in base mesh object

        Example:
        from pylagrit import PyLaGriT
        #create source mesh
        npts = (1,91,1)
        mins = (3.,0.,0.)
        maxs = (3.,270.,0.)
        src_mo = lg.create()
        src_mo.createpts_rtz(npts,mins,maxs,connect=False)

        #create sink mesh
        snk_mo = lg.create()
        snk_mo.createpts_xyz([30,30,1],[-5.,-5.,-5.],[5.,5.,5.],connect=False)

        #compute distance and store in sink mesh attribute 'dfield'
        snk_mo.compute_distance(src_mo)
        snk_mo.dump('comptest.gmv')
        """
        if option not in ["distance_field", "signed_distance_field"]:
            print("ERROR: 'option' must be 'distance_field' or 'signed_distance_field'")
            return

        self.sendcmd("/".join(["compute", option, self.name, mo.name, attname]))

    def compute_extrapolate(self, surf_mo: "MO", dir="zpos", attname="zic"):
        """
        Given a 3D mesh and a 2D surface, this command will extrapolate a scalar
         value from that surface onto every point of the mesh.

        :kwarg surf_mo: Surface mesh object to extrapolate from
        :type  surf_mo: LaGriT mesh object

        :kwarg dir: The direction values are extrapolated from. Choices are one
        of: 'zpos', 'zneg', 'ypos', 'yneg', 'xpos', 'xneg'
        :type  dir: str

        :kwarg attname: The name of the attribute in the surface mesh to be
        extrapolated
        :type  attname: str

        Returns: New attribute in base mesh object

        Example:
        from pylagrit import PyLaGriT
        #create surface mesh
        p1 = (-1.,-1.,-1.)
        p2 = (301.,-1.,-1.)
        p3 = (301.,301.,-1.)
        p4 = (-1.,301.,-1.)
        pts = [p1,p2,p3,p4]
        nnodes = (30,30,1)
        surf = lg.create_qua()
        surf.quadxy(nnodes,pts)

        #make surface mesh interesting
        surf.math('sin','zic',cmosrc=surf,attsrc='xic')
        surf.math('multiply','zic',value=5.0,cmosrc=surf,attsrc='zic')
        surf.perturb(0.,0.,1.)
        surf.math('add','zic',value=60.0,cmosrc=surf,attsrc='zic')

        #create base mesh
        hex = lg.create_hex()
        hex.createpts_brick_xyz([30,30,20],[0.,0.,0.],[300.,300.,50.])
        hex.resetpts_itp()

        #extrapolate z values from surface mesh to base mesh
        hex.compute_extrapolate(surf)
        hex.dump('extrapolated.gmv')
        """

        self.sendcmd(
            "/".join(
                ["compute", "linear_transform", self.name, surf_mo.name, dir, attname]
            )
        )

    def pset_region(
        self, region: "Region", stride=(1, 0, 0), name: Optional[str] = None
    ):
        """
        Define PSet by region

        :kwarg region: region to create pset
        :type  value: PyLaGriT Region object

        :kwarg stride: Nodes defined by ifirst, ilast, and istride.
        :type  stride: list[int, int, int]

        :kwarg name: The name to be assigned to the PSet created.
        :type  name: str

        Returns: PSet object
        """
        if name is None:
            name = make_name("p", self.pset.keys())

        stride = [str(v) for v in stride]

        cmd = "/".join(["pset", name, "region", region.name, ",".join(stride)])
        self.sendcmd(cmd)
        self.pset[name] = PSet(name, self)

        return self.pset[name]

    def pset_surface(
        self, surface: "Surface", stride=(1, 0, 0), name: Optional[str] = None
    ):
        """
        Define PSet by surface

        :kwarg surface: surface to create pset
        :type  value: PyLaGriT Surface object

        :kwarg stride: Nodes defined by ifirst, ilast, and istride.
        :type  stride: list[int, int, int]

        :kwarg name: The name to be assigned to the PSet created.
        :type  name: str

        Returns: PSet object
        """
        if name is None:
            name = make_name("p", self.pset.keys())

        stride = [str(v) for v in stride]

        cmd = "/".join(["pset", name, "surface", surface.name, ",".join(stride)])
        self.sendcmd(cmd)
        self.pset[name] = PSet(name, self)

        return self.pset[name]

    # def pset_not(self, ps, name=None):
    #    '''
    #    Return PSet from Logical Not
    #
    #    Defines and returns a PSet from points that are not inside the PSet, ps.
    #    '''
    #
    #    #Generated a name if one is not specified.
    #    if name is None:
    #        name = make_name('p',self.pset.keys())
    #
    #    #Create the new PSET in lagrit and the pylagrit object.
    #    cmd = 'pset/%s/not/%s'%(name, str(ps))
    #    self.sendline(cmd)
    #    self.pset[name] = PSet(name, self)
    #
    #    return self.pset[name]

    def pset_bool(
        self, pset_list: List["PSet"], boolean="union", name: Optional[str] = None
    ):
        """
        Return PSet from boolean operation on list of psets

        Defines and returns a PSet from points that are not inside the PSet, ps.
        """
        # Generated a name if one is not specified.
        if name is None:
            name = make_name("p", self.pset.keys())

        # Create the new PSET in lagrit and the pylagrit object.
        cmd = ["pset", name, boolean]
        cmd.append(",".join([p.name for p in pset_list]))
        self.sendcmd("/".join(cmd))
        self.pset[name] = PSet(name, self)
        return self.pset[name]

    def pset_union(self, pset_list: List["PSet"], name: Optional[str] = None):
        return self.pset_bool(pset_list, boolean="union", name=name)

    def pset_inter(self, pset_list: List["PSet"], name: Optional[str] = None):
        return self.pset_bool(pset_list, boolean="inter", name=name)

    def pset_not(self, pset_list: List["PSet"], name: Optional[str] = None):
        return self.pset_bool(pset_list, boolean="not", name=name)

    def resetpts_itp(self):
        """
        set node type from connectivity of mesh

        """
        self.sendcmd("resetpts/itp")

    def eltset_object(
        self, mo: "MO", name: Optional[str] = None, attr_name: Optional[str] = None
    ):
        """
        Create element set from the intersecting elements with another mesh object
        """
        if name is None:
            name = make_name("e", self.eltset.keys())
        attr_name = self.intersect_elements(mo, attr_name)
        _ = self.eltset_attribute(attr_name, 0, boolstr="gt")
        self.eltset[name] = EltSet(name, self)
        return self.eltset[name]

    def eltset_bool(
        self, eset_list: List["EltSet"], boolstr="union", name: Optional[str] = None
    ):
        """
        Create element set from boolean operation of set of element sets

        :arg eset_list: List of elements to perform boolean operation on
        :type eset_list: lst(PyLaGriT element set)
        :arg boolstr: type of boolean operation to perform on element sets, one of [union,inter,not]
        :type boolstr: str
        :arg name: The name to be assigned to the EltSet within LaGriT
        :type name: str
        :returns: PyLaGriT element set object
        """
        if name is None:
            name = make_name("e", self.eltset.keys())
        cmd = ["eltset", name, boolstr, " ".join([e.name for e in eset_list])]
        self.sendcmd("/".join(cmd))
        self.eltset[name] = EltSet(name, self)
        return self.eltset[name]

    def eltset_union(self, eset_list: List["EltSet"], name: Optional[str] = None):
        return self.eltset_bool(eset_list, "union", name=name)

    def eltset_inter(self, eset_list: List["EltSet"], name: Optional[str] = None):
        return self.eltset_bool(eset_list, "inter", name=name)

    def eltset_not(self, eset_list: List["EltSet"], name: Optional[str] = None):
        return self.eltset_bool(eset_list, "not", name=name)

    def eltset_region(self, region: "Region", name: Optional[str] = None):
        if name is None:
            name = make_name("e", self.eltset.keys())
        cmd = "/".join(["eltset", name, "region", region.name])
        self.sendcmd(cmd)
        self.eltset[name] = EltSet(name, self)
        return self.eltset[name]

    def eltset_attribute(
        self,
        attribute_name: str,
        attribute_value: int | float,
        boolstr="eq",
        name: Optional[str] = None,
    ):
        if name is None:
            name = make_name("e", self.eltset.keys())
        cmd = "/".join(["eltset", name, attribute_name, boolstr, str(attribute_value)])
        self.sendcmd(cmd)
        self.eltset[name] = EltSet(name, self)
        return self.eltset[name]

    def eltset_write(
        self, filename_root: str, eset_name: Optional["EltSet"] = None, ascii=True
    ):
        """
        Write element set(s) to a file in ascii or binary format

        :arg filename_root: root name of file
        :type filename_root: str
        :arg eset_name: name of eltset to write; if blank, all eltsets in mesh object are written
        :type eset_name: EltSet object
        :arg ascii: Switch to indicate ascii [True] or binary [False]
        :type name: boolean

        Example:
            >>> from pylagrit import PyLaGriT
            >>> import numpy as np
            >>> import sys
            >>>
            >>> lg = PyLaGriT()
            >>>
            >>> dxyz = np.array([0.1, 0.25, 0.25])
            >>> mins = np.array([0.0, 0.0, 0.0])
            >>> maxs = np.array([1.0, 1.0, 1.0])
            >>> mqua = lg.createpts_dxyz(
            ...     dxyz, mins, maxs, "quad", hard_bound=("min", "max", "min"), connect=True
            ... )
            >>>
            >>> example_pset1 = mqua.pset_geom_xyz(mins, maxs - (maxs - mins) / 2)
            >>> example_eset1 = example_pset1.eltset()
            >>> example_pset2 = mqua.pset_geom_xyz(mins + maxs / 2, maxs)
            >>> example_eset2 = example_pset2.eltset()
            >>> # to write one specific eltset
            >>> mqua.eltset_write("test_specific", eset_name=example_eset1)
            >>> # to write all eltsets
            >>> mqua.eltset_write("test_all")
        """
        if eset_name is None:
            name = "-all-"
        else:
            name = eset_name.name
        if ascii is True:
            ascii = "ascii"
        else:
            ascii = "binary"
        cmd = "/".join(["eltset", name, "write", filename_root, ascii])
        self._parent.sendcmd(cmd)

    def rmpoint_pset(
        self, pset: "PSet", itype="exclusive", compress=True, resetpts_itp=True
    ):
        cmd = "rmpoint/pset,get," + pset.name + "/" + itype
        self.sendcmd(cmd)
        if compress:
            self.rmpoint_compress(resetpts_itp=resetpts_itp)

    def rmpoint_eltset(self, eltset: "EltSet", compress=True, resetpts_itp=True):
        cmd = "rmpoint/element/eltset,get," + eltset.name
        self.sendcmd(cmd)
        if compress:
            self.rmpoint_compress(resetpts_itp=resetpts_itp)

    def rmpoint_compress(self, filter_bool=False, resetpts_itp=True):
        """
        remove all marked nodes and correct the itet array

        :param resetpts_itp: set node type from connectivity of mesh
        :type resetpts_itp: bool

        """

        if filter_bool:
            self.sendcmd("filter/1,0,0")
        self.sendcmd("rmpoint/compress")
        if resetpts_itp:
            self.resetpts_itp()

    def reorder_nodes(self, order="ascending", cycle="zic yic xic"):
        self.sendcmd("resetpts itp")
        self.sendcmd("/".join(["sort", self.name, "index", order, "ikey", cycle]))
        self.sendcmd("reorder / " + self.name + " / ikey")
        self.sendcmd("cmo / DELATT / " + self.name + " / ikey")

    def trans(
        self,
        xold: Tuple[float, float, float],
        xnew: Tuple[float, float, float],
        stride=(1, 0, 0),
    ):
        """Translate mesh according to old coordinates "xold" to new coordinates "xnew"

        :param xold: old position
        :type xold: tuple(float,float,float)
        :param xnew: new position
        :type xnew: tuple(float,float,float)
        :param stride: tuple of (first, last, stride) of points
        :type stride: tuple(int,int,int)
        """
        cmd = "/".join(
            [
                "trans",
                ",".join([str(v) for v in stride]),
                ",".join([str(v) for v in xold]),
                ",".join([str(v) for v in xnew]),
            ]
        )
        self.sendcmd(cmd)

    def rotateln(
        self,
        coord1: Tuple[float, float, float],
        coord2: Tuple[float, float, float],
        theta: float,
        center=(0.0, 0.0, 0.0),
        copy=False,
        stride=(1, 0, 0),
    ):
        """
        Rotates a point distribution (specified by ifirst,ilast,istride) about a line.
        The copy option allows the user to make a copy of the original points as well
        as the rotated points, while copy=False just keeps the rotated points themselves.
        The line of rotation defined by coord1 and coord2 needs to be defined such that
        the endpoints extend beyond the point distribution being rotated. theta (in degrees)
        is the angle of rotation whose positive direction is determined by the right-hand-rule,
        that is, if the thumb of your right hand points in the direction of the line
        (1 to 2), then your fingers will curl in the direction of rotation. center is the point
        where the line can be shifted to before rotation takes place.
        If the copy option is chosen, the new points will have only coordinate values
        (xic, yic, zic); no values will be set for any other mesh object attribute for these points.
        Note:  The end points of the  line segment must extend well beyond the point set being rotated.

        Example 1:
            >>> from pylagrit import PyLaGriT
            >>> import numpy
            >>> x = numpy.arange(0, 10.1, 1)
            >>> y = x
            >>> z = [0, 1]
            >>> lg = PyLaGriT()
            >>> mqua = lg.gridder(x, y, z, elem_type="hex", connect=True)
            >>> mqua.rotateln([mqua.xmin - 0.1, 0, 0], [mqua.xmax + 0.1, 0, 0], 25)
            >>> mqua.dump_exo("rotated.exo")
            >>> mqua.dump_ats_xml("rotated.xml", "rotated.exo")
            >>> mqua.paraview()

        Example 2:
            >>> from pylagrit import PyLaGriT
            >>> import numpy
            >>> x = numpy.arange(0, 10.1, 1)
            >>> y = [0, 1]
            >>> # z = [0,1]
            >>> lg = PyLaGriT()
            >>> layer = lg.gridder(x=x, y=y, elem_type="quad", connect=True)
            >>> layer.rotateln([0, layer.ymin - 0.10, 0], [0, layer.ymax + 0.1, 0], 25)
            >>> layer.dump("tmp_lay_top.inp")
            >>> # Layer depths?
            >>> #           1   2   3    4    5    6    7   8    9   10
            >>> layers = [0.1, 1.0]
            >>> addnum = [4, 2]
            >>> # matnum = [2]*len(layers)
            >>> matnum = [2, 1]
            >>> layer_interfaces = numpy.cumsum(layers)
            >>> mtop = layer.copy()
            >>> stack_files = ["tmp_lay_top.inp 1,9"]
            >>> # stack_files.append('tmp_lay_peat_bot.inp 1,33')
            >>> i = 1
            >>> for li,m,a in zip(layer_interfaces,matnum,addnum):
            >>>     layer.math('sub',li,'zic',cmosrc=mtop)
            >>>     stack_files.append('tmp_lay'+str(i)+'.inp '+str(int(m))+', '+str(a))
            >>>     layer.dump('tmp_lay'+str(i)+'.inp')
            >>>     i += 1
            >>> layer.math("sub", 2, "zic", cmosrc=mtop)
            >>> # layer.setatt('zic',-2.)
            >>> layer.dump("tmp_lay_bot.inp")
            >>> stack_files.append("tmp_lay_bot.inp 2")
            >>> stack_files.reverse()
            >>> # Create stacked layer mesh and fill
            >>> stack = lg.create()
            >>> stack.stack_layers("avs", stack_files, flip_opt=True)
            >>> stack_hex = stack.stack_fill()
            >>> stack_hex.dump_exo("rotated.exo")
            >>> stack_hex.dump_ats_xml("rotated.xml", "rotated.exo")
            >>> stack_hex.paraview()
        """
        self.sendcmd(
            "/".join(
                [
                    "rotateln",
                    ",".join([str(v) for v in stride]),
                    "copy" if copy else "nocopy",
                    ",".join([str(v) for v in coord1]),
                    ",".join([str(v) for v in coord2]),
                    str(theta),
                    ",".join([str(v) for v in center]),
                ]
            )
        )

    def massage(
        self,
        bisection_len: float,
        merge_len: float,
        toldamage: float,
        tolroughness: Optional[float] = None,
        stride: Optional[Tuple[int, int, int]] = None,
        nosmooth=False,
        norecon=False,
        strictmergelength=False,
        checkaxy=False,
        semiexclusive=False,
        ignoremats=False,
        lite=False,
    ):
        """
        MASSAGE creates, annihilates, and moves nodes and swaps connections in a 2D or 3D mesh
        in order to improve element aspect ratios and establish user-desired edge lengths.

        The actions of MASSAGE are controlled by values of these four parameters:

            bisection_length  - edge length that will trigger bisection.
            merge_length - edge length that will trigger merging.
            toldamage - maximum grid deformation of interfaces and external boundaries
                        allowed in a single merge, smooth or reconnection event.
            tolroughness - (for 2D surface grids only)  measure of grid roughness
                           (deviation from average surface normal) that triggers refinement.

        The final, optional keywork argument(s) can be one or more of nosmooth, norecon, lite,
        ignoremats, strictmergelength, checkaxy, semiexclusive, and exclusive.

        Specifying nosmooth will turn off the 'smooth' step by skipping the call to SGD.
        Specifying norecon will turn off all 'recon' steps.
        If lite is specified, only one iteration of the merging/reconnection/smoothing
        loop is executed, and a reconnection after edge refinement is omitted.
        This is suitable for applications, such as Gradient Weighted Moving Finite
        Elements, where MASSAGE is called repeatedly.

        The optional argument ignoremats causes MASSAGE to process the multimaterial
        mesh in a single material mode; it ignores the material interfaces.

        The optional argument strictmergelength forces strict interpretation of
        merge_length so that there is no merging along the edges of flat elements.
        This is important if ignoremats is specified to avoid losing the interfaces.

        If checkaxy is given, then we insure that for 2D meshes, the output mesh
        will have positive xy-projected triangle areas, provided that the input mesh
        had them in the first place.

        If exclusive is given, then edge refinement operations will only be performed
        on edges whose endpoints are both in the PSET that MASSAGE is working on.
        (As usual, new nodes created by refinement are added to the PSET so that MASSAGE
        can refine edges recursively.)  The default behavior is 'inclusive',
        where only ONE edge endpoint has to belong to the PSET for the edge to be
        eligible for refinement.

        If semiexclusive is given, refinement will only be triggered by edges with
        both endpoints in the PSET, but some edges with less than two endpoints in
        the PSET might be refined as part of a 'Rivara chain' triggered by the refinement
        of an edge with both endpoints in the PSET.  This represents an intermediate
        case between 'inclusive' and exclusive
        """

        cmd = ["massage", str(bisection_len), str(merge_len), str(toldamage)]

        if tolroughness is not None:
            cmd.append(str(tolroughness))
        if stride is not None:
            cmd.append(",".join([str(x) for x in stride]))

        # Add optional boolean arguments
        _iter = zip(
            [
                "nosmooth",
                "norecon",
                "strictmergelength",
                "checkaxy",
                "semiexclusive",
                "ignoremats",
                "lite",
            ],
            [
                nosmooth,
                norecon,
                strictmergelength,
                checkaxy,
                semiexclusive,
                ignoremats,
                lite,
            ],
        )
        [cmd.append(c[0]) for c in _iter if c[1]]
        self.sendcmd("/".join(cmd))

    def massage2(
        self,
        filename: str,
        min_scale: float,
        bisection_len: float,
        merge_len: float,
        toldamage: float,
        tolroughness: Optional[float] = None,
        stride: Optional[Tuple[int, int, int]] = None,
        nosmooth=False,
        norecon=False,
        strictmergelength=False,
        checkaxy=False,
        semiexclusive=False,
        ignoremats=False,
        lite=False,
    ):
        """
        MASSAGE2 iteratively calls MASSAGE to refine adaptively according to a
        gradient field. Thus, the bisection_length option must be a field.

        file_name is a file which contains a set of LaGriT commands that
        calculates the gradient field based on the distance field. In other
        words, the gradient field is a function of the distance field.
        It is necessary to have this file when using this routine, as the field
        must be updated after each refinement iteration.

        Use this function in conjunction with PyLaGriT.define(**kwargs) for
        best results.

        See MASSAGE for other arguments.
        """

        cmd = [
            "massage2",
            filename,
            str(min_scale),
            str(bisection_len),
            str(merge_len),
            str(toldamage),
        ]
        if tolroughness is not None:
            cmd.append(str(tolroughness))
        if stride is not None:
            cmd.append(",".join([str(x) for x in stride]))

        # Add optional boolean arguments
        _iter = zip(
            [
                "nosmooth",
                "norecon",
                "strictmergelength",
                "checkaxy",
                "semiexclusive",
                "ignoremats",
                "lite",
            ],
            [
                nosmooth,
                norecon,
                strictmergelength,
                checkaxy,
                semiexclusive,
                ignoremats,
                lite,
            ],
        )
        [cmd.append(c[0]) for c in _iter if c[1]]
        self.sendcmd("/".join(cmd))

    def perturb(self, xfactor: float, yfactor: float, zfactor: float, stride=(1, 0, 0)):
        """
        This command moves node coordinates in the following manner.

        Three pairs of random numbers between 0 and 1 are generated.
        These pairs refer to the x, y and z coordinates of the nodes respectively.
        The first random number of each pair is multiplied by the factor given in
        the command. The second random number is used to determine
        if the calculated offset is to be added or subtracted from the coordinate.
        """

        cmd = [
            "perturb",
            ",".join([str(x) for x in stride]),
            str(xfactor),
            str(yfactor),
            str(zfactor),
        ]
        self.sendcmd("/".join(cmd))

    def upscale(
        self,
        method: str,
        attsink: str,
        cmosrc: "MO",
        attsrc: Optional[str] = None,
        stride=(1, 0, 0),
        boundary_choice: Optional[str] = None,
        keepatt=False,
        set_id=False,
    ):
        """
        The upscale command is used to interpolate attribute values from nodes of a fine source mesh to node
        attributes of a coarse sink mesh. The subroutine finds nodes of the fine source mesh within the Voronoi
        cell of every node in the coarser sink mesh. Nodes on cell boundaries are assigned to two or more sink
        nodes. Then the attributes of all the source nodes within a source node's cell are upscaled into a
        single value based on the chosen method. Mesh elements and connectivity are ignored and only node
        values are used to upscale values on to the sink mesh nodes.

        :param method: Type of upscaling: sum, min, max, and averages ariave, harave, geoave
        :type method: str
        :param attsink: attribute sink
        :type attsink: str
        :param cmosrc: PyLaGriT mesh object source
        :type cmosrc: PyLaGriT Mesh Object
        :param attsrc: attribute src, defaults to name of attsink
        :type attsrc: str
        :param stride: tuple of (first, last, stride) of points
        :type stride: tuple(int)
        :param boundary_choice: method of choice when source nodes are found on the boundary of multiple Voronoi volumes of sink nodes: single, divide, or multiple
        :type boundary_choice: str
        """
        if attsrc is None:
            attsrc = attsink
        cmd = [
            "upscale",
            method,
            self.name,
            attsink,
            ",".join([str(v) for v in stride]),
            cmosrc.name,
            attsrc,
        ]
        opts = []
        if boundary_choice is not None:
            opts.append(boundary_choice)
        if keepatt:
            opts.append("keepatt")
        if set_id:
            opts.append("set_id")
        if len(opts) > 0:
            cmd.append(" ".join(opts))
        self.sendcmd("/".join(cmd))

    def upscale_ariave(
        self,
        attsink: str,
        cmosrc: "MO",
        attsrc: Optional[str] = None,
        stride=(1, 0, 0),
        boundary_choice: Optional[str] = None,
        keepatt=False,
        set_id=False,
    ):
        """
        Upscale using arithmetic average of cmosrc points within Voronoi volumes of current mesh

        :param attsink: attribute sink
        :type attsink: str
        :param cmosrc: PyLaGriT mesh object source
        :type cmosrc: PyLaGriT Mesh Object
        :param attsrc: attribute src
        :type attsrc: str
        :param stride: tuple of (first, last, stride) of points
        :type stride: tuple(int)
        :param boundary_choice: method of choice when source nodes are found on the boundary of multiple Voronoi volumes of sink nodes: single, divide, or multiple
        :type boundary_choice: str
        """
        self.upscale(
            "ariave", attsink, cmosrc, attsrc, stride, boundary_choice, keepatt, set_id
        )

    def upscale_geoave(
        self,
        attsink: str,
        cmosrc: "MO",
        attsrc: Optional[str] = None,
        stride=(1, 0, 0),
        boundary_choice: Optional[str] = None,
        keepatt=False,
        set_id=False,
    ):
        """
        Upscale using geometric average of cmosrc points within Voronoi volumes of current mesh

        :param attsink: attribute sink
        :type attsink: str
        :param cmosrc: PyLaGriT mesh object source
        :type cmosrc: PyLaGriT Mesh Object
        :param attsrc: attribute src
        :type attsrc: str
        :param stride: tuple of (first, last, stride) of points
        :type stride: tuple(int)
        :param boundary_choice: method of choice when source nodes are found on the boundary of multiple Voronoi volumes of sink nodes: single, divide, or multiple
        :type boundary_choice: str
        """
        self.upscale(
            "geoave", attsink, cmosrc, attsrc, stride, boundary_choice, keepatt, set_id
        )

    def upscale_harave(
        self,
        attsink: str,
        cmosrc: "MO",
        attsrc: Optional[str] = None,
        stride=(1, 0, 0),
        boundary_choice: Optional[str] = None,
        keepatt=False,
        set_id=False,
    ):
        """
        Upscale using harmonic average of cmosrc points within Voronoi volumes of current mesh

        :param attsink: attribute sink
        :type attsink: str
        :param cmosrc: PyLaGriT mesh object source
        :type cmosrc: PyLaGriT Mesh Object
        :param attsrc: attribute src
        :type attsrc: str
        :param stride: tuple of (first, last, stride) of points
        :type stride: tuple(int)
        :param boundary_choice: method of choice when source nodes are found on the boundary of multiple Voronoi volumes of sink nodes: single, divide, or multiple
        :type boundary_choice: str
        """
        self.upscale(
            "harave", attsink, cmosrc, attsrc, stride, boundary_choice, keepatt, set_id
        )

    def upscale_min(
        self,
        attsink: str,
        cmosrc: "MO",
        attsrc: Optional[str] = None,
        stride=(1, 0, 0),
        boundary_choice: Optional[str] = None,
        keepatt=False,
        set_id=False,
    ):
        """
        Upscale using minimum of cmosrc points within Voronoi volumes of current mesh

        :param attsink: attribute sink
        :type attsink: str
        :param cmosrc: PyLaGriT mesh object source
        :type cmosrc: PyLaGriT Mesh Object
        :param attsrc: attribute src
        :type attsrc: str
        :param stride: tuple of (first, last, stride) of points
        :type stride: tuple(int)
        :param boundary_choice: method of choice when source nodes are found on the boundary of multiple Voronoi volumes of sink nodes: single, divide, or multiple
        :type boundary_choice: str
        """
        self.upscale(
            "min", attsink, cmosrc, attsrc, stride, boundary_choice, keepatt, set_id
        )

    def upscale_max(
        self,
        attsink: str,
        cmosrc: "MO",
        attsrc: Optional[str] = None,
        stride=(1, 0, 0),
        boundary_choice: Optional[str] = None,
        keepatt=False,
        set_id=False,
    ):
        """
        Upscale using maximum of cmosrc points within Voronoi volumes of current mesh

        :param attsink: attribute sink
        :type attsink: str
        :param cmosrc: PyLaGriT mesh object source
        :type cmosrc: PyLaGriT Mesh Object
        :param attsrc: attribute src
        :type attsrc: str
        :param stride: tuple of (first, last, stride) of points
        :type stride: tuple(int)
        :param boundary_choice: method of choice when source nodes are found on the boundary of multiple Voronoi volumes of sink nodes: single, divide, or multiple
        :type boundary_choice: str
        """
        self.upscale(
            "max", attsink, cmosrc, attsrc, stride, boundary_choice, keepatt, set_id
        )

    def upscale_sum(
        self,
        attsink: str,
        cmosrc: "MO",
        attsrc: Optional[str] = None,
        stride=(1, 0, 0),
        boundary_choice: Optional[str] = None,
        keepatt=False,
        set_id=False,
    ):
        """
        Upscale using sum of cmosrc points within Voronoi volumes of current mesh

        :param attsink: attribute sink
        :type attsink: str
        :param cmosrc: PyLaGriT mesh object source
        :type cmosrc: PyLaGriT Mesh Object
        :param attsrc: attribute src
        :type attsrc: str
        :param stride: tuple of (first, last, stride) of points
        :type stride: tuple(int)
        :param boundary_choice: method of choice when source nodes are found on the boundary of multiple Voronoi volumes of sink nodes: single, divide, or multiple
        :type boundary_choice: str
        """
        self.upscale(
            "sum", attsink, cmosrc, attsrc, stride, boundary_choice, keepatt, set_id
        )

    def gmv(self, exe: Optional[str] = None, filename: Optional[str] = None):
        if filename is None:
            filename = self.name + ".gmv"
        if exe is not None:
            self._parent.gmv_exe = exe
        self.sendcmd("dump/gmv/" + filename + "/" + self.name)
        os.system(self._parent.gmv_exe + " -i " + filename)  # noqa: S605

    def paraview(self, exe: Optional[str] = None, filename: Optional[str] = None):
        if filename is None:
            filename = self.name + ".inp"
        if exe is not None:
            self._parent.paraview_exe = exe
        self.sendcmd("dump/avs/" + filename + "/" + self.name)
        os.system(self._parent.paraview_exe + " " + filename)  # noqa: S605

    def dump(self, filename: Optional[str] = None, format: Optional[str] = None, *args):
        if filename is None and format is None:
            print("Error: At least one of either filename or format option is required")
            return
        # if format is not None: cmd = '/'.join(['dump',format])
        # else: cmd = 'dump'
        if filename is not None and format is not None:
            if format in ["fehm", "zone_outside", "zone_outside_minmax"]:
                filename = filename.split(".")[0]
            if format == "stor" and len(args) == 0:
                filename = filename.split(".")[0]
            cmd = "/".join(["dump", format, filename, self.name])
        elif format is not None:
            if format in ["avs", "avs2"]:
                filename = self.name + ".inp"
            elif format == "fehm":
                filename = self.name
            elif format == "gmv":
                filename = self.name + ".gmv"
            elif format == "tecplot":
                filename = self.name + ".plt"
            elif format == "lagrit":
                filename = self.name + ".lg"
            elif format == "exo":
                filename = self.name + ".exo"
            else:
                raise NotImplementedError("Unsupported format")
            cmd = "/".join(["dump", format, filename, self.name])
        elif filename is not None:
            cmd = "/".join(["dump", filename, self.name])
        else:
            cmd = "/".join(["dump", self.name + ".inp"])

        for arg in args:
            cmd = "/".join([cmd, str(arg)])

        self.sendcmd(cmd)

    def dump_avs2(
        self,
        filename: str,
        points=True,
        elements=True,
        node_attr=True,
        element_attr=True,
    ):
        """
        Dump avs file

        :arg filename: Name of avs file
        :type filename: str
        :arg points: Output point coordinates
        :type points: bool
        :arg elements: Output connectivity
        :type elements: bool
        :arg node_attr: Output node attributes
        :type node_attr: bool
        :arg element_attr: Output element attributes
        :type element_attr: bool
        """
        self.dump(
            filename,
            "avs2",
            int(points),
            int(elements),
            int(node_attr),
            int(element_attr),
        )

    def dump_exo(
        self,
        filename: str,
        psets=False,
        eltsets=False,
        facesets: Optional[List["FaceSet"]] = None,
    ):
        """
        Dump exo file

        :arg filename: Name of exo file
        :type filename: str
        :arg psets: Boolean indicating that exodus will only include psets
        :type psets: bool
        :arg eltsets: Boolean indicating that exodus will only include element sets
        :type eltsets: bool
        :arg facesets:  Array of FaceSet objects
        :type facesets: lst(FaceSet)

        Example:
            >>> from pylagrit import PyLaGriT
            >>> l = PyLaGriT()
            >>> m = l.create()
            >>> m.createpts_xyz(
            ...     (3, 3, 3),
            ...     (0.0, 0.0, 0.0),
            ...     (1.0, 1.0, 1.0),
            ...     rz_switch=[1, 1, 1],
            ...     connect=True,
            ... )
            >>> m.status()
            >>> m.status(brief=True)
            >>> fs = m.create_boundary_facesets(base_name="faceset_bounds")
            >>> m.dump_exo("cube.exo", facesets=fs.values())
        """
        cmd = "/".join(["dump/exo", filename, self.name])
        if psets:
            cmd = "/".join([cmd, "psets"])
        else:
            cmd = "/".join([cmd, " "])
        if eltsets:
            cmd = "/".join([cmd, "eltsets"])
        else:
            cmd = "/".join([cmd, " "])
        if facesets is not None:
            cmd = "/".join([cmd, "facesets"])
            for fc in facesets:
                cmd += " &\n" + fc.filename
        self.sendcmd(cmd)

    def dump_gmv(self, filename: str, format="binary"):
        self.dump(filename, "gmv", format)

    def dump_fehm(self, filename: str, *args):
        self.dump(filename, "fehm", *args)

    def dump_lg(self, filename: str, format="binary"):
        self.dump(filename, "lagrit", format)

    def dump_zone_imt(self, filename: str, imt_value: int):
        cmd = ["dump", "zone_imt", filename, self.name, str(imt_value)]
        self.sendcmd("/".join(cmd))

    def dump_pflotran(self, filename_root: str, nofilter_zero=False):
        """
        Dump PFLOTRAN UGE file

        :arg filename_root: root name of UGE file
        :type filename_root: str
        :arg nofilter_zero:  Set to true to write zero coefficients to file
        :type nofilter_zero: boolean

        Example:
            >>> from pylagrit import PyLaGriT
            >>> l = PyLaGriT()
            >>> m = l.create()
            >>> m.createpts_xyz(
            ...     (3, 3, 3),
            ...     (0.0, 0.0, 0.0),
            ...     (1.0, 1.0, 1.0),
            ...     rz_switch=[1, 1, 1],
            ...     connect=True,
            ... )
            >>> m.status()
            >>> m.status(brief=True)
            >>> m.dump_pflotran("test_pflotran_dump")
        """
        cmd = ["dump", "pflotran", filename_root, self.name]
        if nofilter_zero:
            cmd.append("nofilter_zero")
        self.sendcmd("/".join(cmd))

    def dump_zone_outside(
        self, filename: str, keepatt=False, keepatt_median=False, keepatt_voronoi=False
    ):
        cmd = ["dump", "zone_outside", filename, self.name]
        if keepatt:
            cmd.append("keepatt")
        if keepatt_median and keepatt_voronoi:
            print("Error: keepatt_median and keepatt_voronoi cannot both be True")
            return
        elif keepatt_median:
            cmd.append("keepatt_median")
        elif keepatt_voronoi:
            cmd.append("keepatt_voronoi")
        self.sendcmd("/".join(cmd))

    def dump_ats_xml(
        self,
        filename: str,
        meshfilename: str,
        matnames: Dict[str, int] = {},  # noqa: B006
        facenames: Dict[str, int] = {},  # noqa: B006
    ):
        """
        Write ats style xml file with regions
        :param filename: Name of xml to write
        :type filename: string
        :param meshfilename: Name of exodus file to use in xml
        :type meshfilename: string
        :param matnames: Dictionary of region names keyed by exodus material number
        :type matnames: dict
        :param facenames: Dictionary of faceset names keyed by exodus faceset number
        :type facenames: dict
        """
        main = ET.Element("ParameterList", {"name": "Main", "type": "ParameterList"})

        ET.SubElement(
            main,
            "Parameter",
            {"name": "Native Unstructured Input", "type": "bool", "value": "true"},
        )
        ET.SubElement(
            main,
            "Parameter",
            {"name": "grid_option", "type": "string", "value": "Unstructured"},
        )

        mesh = ET.SubElement(
            main, "ParameterList", {"name": "Mesh", "type": "ParameterList"}
        )
        ET.SubElement(
            mesh,
            "Parameter",
            {"isUsed": "true", "name": "Framework", "type": "string", "value": "MSTK"},
        )

        mesh1 = ET.SubElement(
            mesh, "ParameterList", {"name": "Read Mesh File", "type": "ParameterList"}
        )
        ET.SubElement(
            mesh1,
            "Parameter",
            {"name": "File", "type": "string", "value": meshfilename},
        )
        ET.SubElement(
            mesh1,
            "Parameter",
            {"name": "Format", "type": "string", "value": "Exodus II"},
        )

        mesh2 = ET.SubElement(
            mesh, "ParameterList", {"name": "Surface Mesh", "type": "ParameterList"}
        )
        ET.SubElement(
            mesh2,
            "Parameter",
            {"name": "surface sideset name", "type": "string", "value": "surface"},
        )
        mesh2a = ET.SubElement(
            mesh2, "ParameterList", {"name": "Expert", "type": "ParameterList"}
        )
        ET.SubElement(
            mesh2a,
            "Parameter",
            {"name": "Verify Mesh", "type": "bool", "value": "false"},
        )

        r = ET.SubElement(
            main, "ParameterList", {"name": "Regions", "type": "ParameterList"}
        )

        r1 = ET.SubElement(
            r,
            "ParameterList",
            {"name": "computational domain", "type": "ParameterList"},
        )
        l1 = ET.SubElement(
            r1, "ParameterList", {"name": "Region: Box", "type": "ParameterList"}
        )
        ET.SubElement(
            l1,
            "Parameter",
            {
                "name": "Low Coordinate",
                "type": "Array(double)",
                "value": "{-1.e20,-1.e20,-1.e20}",
            },
        )
        ET.SubElement(
            l1,
            "Parameter",
            {
                "name": "High Coordinate",
                "type": "Array(double)",
                "value": "{1.e20,1.e20,1.e20}",
            },
        )

        r2 = ET.SubElement(
            r, "ParameterList", {"name": "surface domain", "type": "ParameterList"}
        )
        l2 = ET.SubElement(
            r2, "ParameterList", {"name": "Region: Box", "type": "ParameterList"}
        )
        ET.SubElement(
            l2,
            "Parameter",
            {
                "name": "Low Coordinate",
                "type": "Array(double)",
                "value": "{-1.e20,-1.e20}",
            },
        )
        ET.SubElement(
            l2,
            "Parameter",
            {
                "name": "High Coordinate",
                "type": "Array(double)",
                "value": "{1.e20,1.e20}",
            },
        )

        rmat = []
        lmat = []
        for k, v in matnames.items():
            rmat.append(
                ET.SubElement(
                    r, "ParameterList", {"name": str(v), "type": "ParameterList"}
                )
            )
            lmat.append(
                ET.SubElement(
                    rmat[-1],
                    "ParameterList",
                    {"name": "Region: Labeled Set", "type": "ParameterList"},
                )
            )
            ET.SubElement(
                lmat[-1],
                "Parameter",
                {"name": "Label", "type": "string", "value": str(k)},
            )
            ET.SubElement(
                lmat[-1],
                "Parameter",
                {"name": "File", "type": "string", "value": meshfilename},
            )
            ET.SubElement(
                lmat[-1],
                "Parameter",
                {"name": "Format", "type": "string", "value": "Exodus II"},
            )
            ET.SubElement(
                lmat[-1],
                "Parameter",
                {"name": "Entity", "type": "string", "value": "Cell"},
            )

        rsurf = []
        lsurf = []
        for k, v in facenames.items():
            rsurf.append(
                ET.SubElement(
                    r, "ParameterList", {"name": str(v), "type": "ParameterList"}
                )
            )
            lsurf.append(
                ET.SubElement(
                    rsurf[-1],
                    "ParameterList",
                    {"name": "Region: Labeled Set", "type": "ParameterList"},
                )
            )
            ET.SubElement(
                lsurf[-1],
                "Parameter",
                {"name": "Label", "type": "string", "value": str(k)},
            )
            ET.SubElement(
                lsurf[-1],
                "Parameter",
                {"name": "File", "type": "string", "value": meshfilename},
            )
            ET.SubElement(
                lsurf[-1],
                "Parameter",
                {"name": "Format", "type": "string", "value": "Exodus II"},
            )
            ET.SubElement(
                lsurf[-1],
                "Parameter",
                {"name": "Entity", "type": "string", "value": "Face"},
            )

        m_str = ET.tostring(main)
        m_reparsed = minidom.parseString(m_str)  # noqa: S318
        with open(filename, "w") as f:
            f.write(m_reparsed.toprettyxml(indent="  "))

    def dump_pset(self, filerootname: str, zonetype="zone", pset: List["PSet"] = []):  # noqa: B006
        """
        Dump zone file of psets
        :arg filerootname: rootname of files to create, pset name will be added to name
        :type filerootname: string
        :arg zonetype: Type of zone file to dump, 'zone' or 'zonn'
        :type zonetype: string
        :arg pset: list of psets to dump, all psets dumped if empty list
        :type pset: list[strings]
        """
        if len(pset) == 0:
            cmd = ["pset", "-all-", zonetype, filerootname, "ascii"]
            self.sendcmd("/".join(cmd))
        else:
            for p in pset:
                cmd = ["pset", p.name, zonetype, filerootname + "_" + p.name, "ascii"]
                self.sendcmd("/".join(cmd))

    def delete(self):
        self.sendcmd("cmo/delete/" + self.name)
        del self._parent.mo[self.name]

    def create_boundary_facesets(
        self,
        stacked_layers=False,
        base_name: Optional[str] = None,
        reorder=False,
        external=True,
    ):
        """
        Creates facesets for each boundary and writes associated avs faceset file
        :arg base_name: base name of faceset files
        :type base_name: str
        :arg stacked_layers: if mesh is created by stack_layers, user layertyp attr to determine top and bottom
        :type stacked_layers: bool
        :arg reorder_on_meds: reorder nodes on cell medians, usually needed for exodus file
        :type reorder_on_meds: bool
        :returns: Dictionary of facesets
        """
        if base_name is None:
            base_name = "faceset_" + self.name
        mo_surf = self.extract_surfmesh(reorder=reorder, external=external)
        mo_surf.addatt("id_side", vtype="vint", rank="scalar", length="nelements")
        mo_surf.settets_normal()
        mo_surf.copyatt("itetclr", "id_side")
        mo_surf.delatt(["id_side"])
        fs = OrderedDict()
        if stacked_layers:
            pbot = mo_surf.pset_attribute("layertyp", -1)
            ebot = pbot.eltset(membership="exclusive")
        else:
            ebot = mo_surf.eltset_attribute("itetclr", 1)
        fs["bottom"] = ebot.create_faceset(base_name + "_bottom.avs")
        if stacked_layers:
            ptop = mo_surf.pset_attribute("layertyp", -2)
            etop = ptop.eltset(membership="exclusive")
        else:
            etop = mo_surf.eltset_attribute("itetclr", 2)
        fs["top"] = etop.create_faceset(base_name + "_top.avs")
        er = mo_surf.eltset_attribute("itetclr", 3)
        fs["right"] = er.create_faceset(base_name + "_right.avs")
        eback = mo_surf.eltset_attribute("itetclr", 4)
        fs["back"] = eback.create_faceset(base_name + "_back.avs")
        el = mo_surf.eltset_attribute("itetclr", 5)
        fs["left"] = el.create_faceset(base_name + "_left.avs")
        ef = mo_surf.eltset_attribute("itetclr", 6)
        fs["front"] = ef.create_faceset(base_name + "_front.avs")
        return fs

    def createpts(
        self,
        crd: str,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_value=(1, 1, 1),
        connect=False,
    ):
        """
        Create and Connect Points

        :arg crd: Coordinate type of either 'xyz' (cartesian coordinates),
                    'rtz' (cylindrical coordinates), or
                    'rtp' (spherical coordinates).
        :type  crd: str
        :arg  npts: The number of points to create in line
        :type npts: tuple(int)
        :arg  mins: The starting value for each dimension.
        :type mins: tuple(int, int, int)
        :arg  maxs: The ending value for each dimension.
        :type maxs: tuple(int, int, int)
        :kwarg vc_switch: Determines if nodes represent vertices (1) or cell centers (0).
        :type  vc_switch: tuple(int, int, int)
        :kwarg rz_switch: Determines true or false (1 or 0) for using ratio zoning values.
        :type  rz_switch: tuple(int, int, int)

        """

        cmd = "/".join(
            [
                "createpts",
                crd,
                ",".join([str(v) for v in npts]),
                ",".join([str(v) for v in mins]),
                ",".join([str(v) for v in maxs]),
                ",".join([str(v) for v in vc_switch]),
                ",".join([str(v) for v in rz_switch]),
                ",".join([str(v) for v in rz_value]),
            ]
        )
        self.sendcmd(cmd)

        if connect:
            if self.elem_type.startswith(("tri", "tet")):
                cmd = "/".join(["connect", "noadd"])
            else:
                cmd = "/".join(
                    [
                        "createpts",
                        "brick",
                        crd,
                        ",".join([str(v) for v in npts]),
                        "1,0,0",
                        "connect",
                    ]
                )
            self.sendcmd(cmd)

    def createpts_xyz(
        self,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_value=(1, 1, 1),
        connect=True,
    ):
        self.createpts(
            "xyz", npts, mins, maxs, vc_switch, rz_switch, rz_value, connect=connect
        )

    def createpts_dxyz(
        self,
        dxyz: Tuple[float, float, float],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        clip: str | Tuple[str, str, str] = "under",
        hard_bound: str | Tuple[str, str, str] = "min",
        rz_switch=(1, 1, 1),
        rz_value=(1, 1, 1),
        connect=True,
    ):
        """
        Create and Connect Points to create an orthogonal hexahedral mesh. The
        vertex spacing is based on dxyz and the mins and maxs specified. mins
        (default, see hard_bound option) or maxs will be adhered to, while maxs
        (default) or mins will be modified based on the clip option to be
        truncated at the nearest value 'under' (default) or 'over' the range
        maxs-mins. clip and hard_bound options can be mixed by specifying tuples
        (see description below).

        :arg  dxyz: The spacing between points in x, y, and z directions
        :type dxyz: tuple(float,float,float)
        :arg  mins: The starting value for each dimension.
        :type mins: tuple(float,float,float)
        :arg  maxs: The ending value for each dimension.
        :type maxs: tuple(float,float,float)
        :kwarg clip: How to handle bounds if range does not divide by dxyz, either clip 'under' or 'over' range
        :type clip: string or tuple(string,string,string)
        :kwarg hard_bound: Whether to use the "min" or "max" as the hard constraint on dimension
        :type hard_bound: string or tuple(string,string,string)
        :kwarg rz_switch: Determines true or false (1 or 0) for using ratio zoning values.
        :type  rz_switch: tuple(int, int, int)
        :kwarg connect: Whether or not to connect points
        :type  connect: boolean

        """
        if isinstance(hard_bound, str):
            hard_bound = (hard_bound, hard_bound, hard_bound)
        if isinstance(clip, str):
            clips = [clip, clip, clip]
        else:
            clips = clip
        dxyz_ar = numpy.array(dxyz)
        mins_ar = numpy.array(mins)
        maxs_ar = numpy.array(maxs)
        dxyz_ar[dxyz_ar == 0] = 1
        npts = numpy.zeros_like(dxyz_ar).astype("int")
        for i, cl in enumerate(clips):
            if cl == "under":
                npts[i] = int(numpy.floor((maxs_ar[i] - mins_ar[i]) / dxyz_ar[i]))
            elif cl == "over":
                npts[i] = int(numpy.ceil((maxs_ar[i] - mins_ar[i]) / dxyz_ar[i]))
            else:
                print("Error: unrecognized clip option")
                return
        for i, bnd in enumerate(hard_bound):
            if bnd == "min":
                maxs_ar[i] = mins_ar[i] + npts[i] * dxyz_ar[i]
            elif bnd == "max":
                mins_ar[i] = maxs_ar[i] - npts[i] * dxyz_ar[i]
            else:
                print("Error: unrecognized hard_bound option")
                return
        npts += 1
        vc_switch = (1, 1, 1)  # always vertex nodes for dxyz method
        self.createpts(
            "xyz",
            npts.tolist(),
            mins_ar.tolist(),
            maxs_ar.tolist(),
            vc_switch,
            rz_switch,
            rz_value,
            connect=connect,
        )
        if self._parent.verbose:
            self.minmax_xyz()

    def createpts_rtz(
        self,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_value=(1, 1, 1),
        connect=True,
    ):
        self.createpts(
            "rtz", npts, mins, maxs, vc_switch, rz_switch, rz_value, connect=connect
        )

    def createpts_rtp(
        self,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_value=(1, 1, 1),
        connect=True,
    ):
        self.createpts(
            "rtp", npts, mins, maxs, vc_switch, rz_switch, rz_value, connect=connect
        )

    def createpts_line(
        self,
        npts: int,
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
    ):
        """
        Create and Connect Points in a line

        :arg  npts: The number of points to create in line
        :type npts: int
        :arg  mins: The starting value for each dimension.
        :type mins: tuple(int, int, int)
        :arg  maxs: The ending value for each dimension.
        :type maxs: tuple(int, int, int)
        :kwarg vc_switch: Determines if nodes represent vertices (1) or cell centers (0).
        :type  vc_switch: tuple(int, int, int)
        :kwarg rz_switch: Determines true or false (1 or 0) for using ratio zoning values.
        :type  rz_switch: tuple(int, int, int)

        """

        cmd = "/".join(
            [
                "createpts",
                "line",
                str(npts),
                " ",
                " ",
                ",".join([str(v) for v in mins + maxs]),
                ",".join([str(v) for v in vc_switch]),
                ",".join([str(v) for v in rz_switch]),
            ]
        )
        self.sendcmd(cmd)

    def createpts_brick(
        self,
        crd: str,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_vls=(1, 1, 1),
    ):
        """
        Create and Connect Points

        Creates a grid of points in the mesh object and connects them.

        :arg crd: Coordinate type of either 'xyz' (cartesian coordinates),
                    'rtz' (cylindrical coordinates), or
                    'rtp' (spherical coordinates).
        :type  crd: str

        :arg  npts: The number of points to create in each dimension.
        :type npts: tuple(int, int, int)

        :arg  mins: The starting value for each dimension.
        :type mins: tuple(int, int, int)

        :arg  maxs: The ending value for each dimension.
        :type maxs: tuple(int, int, int)

        :kwarg vc_switch: Determines if nodes represent vertices (1) or cell centers (0).
        :type  vc_switch: tuple(int, int, int)

        :kwarg rz_switch: Determines true or false (1 or 0) for using ratio
                          zmoning values.
        :type  rz_switch: tuple(int, int, int)

        :kwarg rz_vls: Ratio zoning values. Each point will be multiplied by
                       a scale of the value for that dimension.
        :type  rz_vls: tuple(int, int, int)
        """

        ni, nj, nk = map(str, npts)
        xmn, ymn, zmn = map(str, mins)
        xmx, ymx, zmx = map(str, maxs)
        iiz, ijz, ikz = map(str, vc_switch)
        iirat, ijrat, ikrat = map(str, rz_switch)
        xrz, yrz, zrz = map(str, rz_vls)

        t = (crd, ni, nj, nk, xmn, ymn, zmn, xmx, ymx, zmx)
        t = t + (iiz, ijz, ikz, iirat, ijrat, ikrat, xrz, yrz, zrz)
        cmd = (
            "createpts/brick/%s/%s,%s,%s/%s,%s,%s/%s,%s,%s/%s,%s,%s/"
            + "%s,%s,%s/%s,%s,%s"
        )
        self.sendcmd(cmd % t)

    def createpts_brick_xyz(
        self,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_vls=(1, 1, 1),
    ):
        """Create and connect Cartesian coordinate points."""
        self.createpts_brick("xyz", **minus_self(locals()))

    def createpts_brick_rtz(
        self,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_vls=(1, 1, 1),
    ):
        """Create and connect cylindrical coordinate points."""
        self.createpts_brick("rtz", **minus_self(locals()))

    def createpts_brick_rtp(
        self,
        npts: Tuple[int, int, int],
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        vc_switch=(1, 1, 1),
        rz_switch=(1, 1, 1),
        rz_vls=(1, 1, 1),
    ):
        """Create and connect spherical coordinates."""
        self.createpts_brick("rtp", **minus_self(locals()))

    def createpts_median(self):
        self.sendcmd("createpts/median")

    def subset(
        self,
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        geom="xyz",
    ):
        """
        Return Mesh Object Subset

        Creates a new mesh object that contains only a geometric subset defined
        by mins and maxs.

        :arg  mins: Coordinate of one of the shape's defining points.
                     xyz (Cartesian):   (x1, y1, z1);
                     rtz (Cylindrical): (radius1, theta1, z1);
                     rtp (Spherical):   (radius1, theta1, phi1);
        :typep mins: tuple(int, int, int)

        :arg  maxs: Coordinate of one of the shape's defining points.
                     xyz (Cartesian):   (x2, y2, z2);
                     rtz (Cylindrical): (radius2, theta2, z2);
                     rtp (Spherical):   (radius2, theta2, phi2);
        :type maxs: tuple(int, int, int)

        :kwarg geom: Type of geometric shape: 'xyz' (spherical),
                     'rtz' (cylindrical), 'rtp' (spherical)
        :type  geom: str

        Returns: MO object

        Example:
            >>> # To use pylagrit, import the module.
            >>> import pylagrit

            >>> # Start the lagrit session.
            >>> lg = pylagrit.PyLaGriT()

            >>> # Create a mesh object.
            >>> mo = lg.create()
            >>> mo.createpts_brick_xyz((5, 5, 5), (0, 0, 0), (5, 5, 5))

            >>> # Take the subset from (3,3,3)
            >>> mo.subset((3, 3, 3), (5, 5, 5))

        """

        lg = self._parent
        new_mo = lg.copy(self)
        sub_pts = new_mo.pset_geom(mins, maxs, geom=geom)
        rm_pts = new_mo.pset_not([sub_pts])

        new_mo.rmpoint_pset(rm_pts)
        return new_mo

    def subset_xyz(
        self, mins: Tuple[float, float, float], maxs: Tuple[float, float, float]
    ):
        """
        Return Tetrehedral MO Subset

        Creates a new mesh object that contains only a tetrehedral subset
        defined by mins and maxs.

        :arg  mins: Coordinate point of 1 of the tetrahedral's corners.
        :type mins: tuple(int, int, int)

        :arg  maxs: Coordinate point of 1 of the tetrahedral's corners.
        :type maxs: tuple(int, int, int)

        Returns: MO object
        """
        return self.subset(geom="xyz", **minus_self(locals()))

    def quadxy(
        self,
        nnodes: Tuple[int, int, int],
        pts: List[Tuple[float, float, float]],
        connect=True,
    ):
        """
        Define an arbitrary, logical quad of points in 3D space
        with nnodes(x,y,z) nodes. By default, the nodes will be connected.

        :arg nnodes: The number of nodes to create in each dimension.
                      One value must == 1 and the other two must be > 1.
        :type nnodes: tuple(int, int, int)

        :arg pts: The four corners of the quad surface, defined in counter
                   clockwise order (the normal to the quad points is defined
                   using the right hand rule and the order of the points).
        :type pts:  list of four 3-tuples (float)

        :arg connect: connect points
        :type connect: bool

        Example:
            >>> # To use pylagrit, import the module.
            >>> import pylagrit

            >>> # Start the lagrit session.
            >>> lg = pylagrit.PyLaGriT()

            >>> # Create a mesh object.
            >>> qua = lg.create_qua()

            >>> # Define 4 points in correct order
            >>> p1 = (0.0, 200.0, -400.0)
            >>> p2 = (0.0, -200.0, -400.0)
            >>> p3 = (140.0, -200.0, 0.0)
            >>> p4 = (118.0, 200.0, 0.0)
            >>> pts = [p1, p2, p3, p4]

            >>> # Define nnodes
            >>> nnodes = (29, 1, 82)

            >>> # Create and connect skewed plane
            >>> qua.quadxy(nnodes, pts)

        """
        self.select()
        quadpts = [n for n in nnodes if n != 1]
        assert len(quadpts) == 2, "nnodes must have one value == 1 and two values > 1"  # noqa: S101

        c = ""
        for v in pts:
            assert len(v) == 3, "vectors must be of length 3 (x,y,z)"  # noqa: S101
            c += "/&\n" + ",".join(list(map(str, v)))
        self.sendcmd("quadxy/%d,%d%s" % (quadpts[0], quadpts[1], c))

        if connect:
            cmd = "/".join(
                [
                    "createpts",
                    "brick",
                    "xyz",
                    ",".join(map(str, nnodes)),
                    "1,0,0",
                    "connect",
                ]
            )
            self.sendcmd(cmd)

    def quadxyz(
        self,
        nnodes: Tuple[int, int, int],
        pts: List[Tuple[float, float, float]],
        connect=True,
    ):
        """
         Define an arbitrary and logical set of points in 3D (xyz) space.
         The set of points will be connected into hexahedrons by default. Set 'connect=False' to prevent connection.

         :arg nnodes: The number of nodes including the 1st and last point along each X, Y, Z axis. The number of points will be 1 more than the number of elements in each dimension.
         :type nnodes: tuple(int, int, int)

         :arg pts: The eight corners of the hexahedron. The four bottom corners are listed first,
         then the four top corners. Each set of corners (bottom and top) are defined in counter-clockwise
         order (the normal to the quad points is defined using the right hand rule and the order of the points).
         :arg pts:  list of eight 3-tuples (float)

         :arg connect: connect points
         :type connect: bool

        Example:
             >>> # To use pylagrit, import the module.
             >>> import pylagrit

             >>> # Start the lagrit session.
             >>> lg = pylagrit.PyLaGriT()

             >>> # Create a mesh object.
             >>> hex = lg.create()

             >>> # Define 4 bottom points in correct order
             >>> p1 = (0.0, 0.0, 0.0)
             >>> p2 = (1.0, 0.0, 0.02)
             >>> p3 = (1.0, 1.0, 0.0)
             >>> p4 = (0.0, 1.0, 0.1)

             >>> # Define 4 top points in correct order
             >>> p5 = (0.0, 0.0, 1.0)
             >>> p6 = (1.0, 0.0, 1.0)
             >>> p7 = (1.0, 1.0, 1.0)
             >>> p8 = (0.0, 1.0, 1.1)

             >>> pts = [p1, p2, p3, p4, p5, p6, p7, p8]

             >>> # Define nnodes
             >>> nnodes = (3, 3, 3)

             >>> # Create and connect skewed hex mesh
             >>> hex.quadxyz(nnodes, pts)
             >>> # Dump mesh
             >>> hex.dump("quadxyz_test.gmv")

        """
        self.select()
        assert len(nnodes) == 3, "nnodes must contain three values"  # noqa: S101
        assert len(pts) == 8, "pts must contain eight sets of points"  # noqa: S101
        cmd = "/".join(["quadxyz", ",".join(map(str, nnodes))])
        for v in pts:
            assert len(v) == 3, "each entry in pts must contain 3 (x,y,z) values"  # noqa: S101
            cmd += "/ &\n" + ",".join(list(map(str, v)))
        self.sendcmd(cmd)

        if connect:
            cmd = "/".join(
                [
                    "createpts",
                    "brick",
                    "xyz",
                    ",".join(map(str, nnodes)),
                    "1,0,0",
                    "connect",
                ]
            )
            self.sendcmd(cmd)

    def rzbrick(
        self,
        n_ijk: Tuple[int, int, int],
        connect=True,
        stride=(1, 0, 0),
        coordinate_space="xyz",
    ):
        """
        Builds a brick mesh and generates a nearest neighbor connectivity matrix

        Currently only configured for this flavor of syntax:

            rzbrick/xyz|rtz|rtp/ni,nj,nk/pset,get,name/connect/

        Use this option with quadxyz to connect logically rectangular grids.

        :arg n_ijk: number of points to be created in each direction.
        :type n_ijk: tuple
        :arg connect: connect points
        :type connect: bool
        :arg stride: Stride to select
        :type stride: tuple
        :arg coordinate_space: xyz,rtz,or rtp coordinate spaces
        :type coordinate_space: str
        """

        assert coordinate_space in ["xyz", "rtz", "rtp"], "Unknown coordinate space"  # noqa: S101

        self.select()
        cmd = f"rzbrick/{coordinate_space}"

        for v in [n_ijk, stride]:
            cmd += "/" + ",".join(list(map(str, v)))

        if connect:
            cmd += "/connect"

        self.sendcmd(cmd)

    def subset_rtz(
        self, mins: Tuple[float, float, float], maxs: Tuple[float, float, float]
    ):
        """
        Return Cylindrical MO Subset

        Creates a new mesh object that contains only a cylindrical subset
        defined by mins and maxs.

        :arg  mins: Defines radius1, theta1, and z1.
        :type mins: tuple(int, int, int)

        :arg  maxs: Defines radius2, theta2, and z2.
        :type maxs: tuple(int, int, int)

        Returns: MO object
        """
        return self.subset(geom="rtz", **minus_self(locals()))

    def subset_rtp(
        self, mins: Tuple[float, float, float], maxs: Tuple[float, float, float]
    ):
        """
        Return Spherical MO Subset

        Creates a new mesh object that contains only a spherical subset
        defined by mins and maxs.

        :arg  mins: Defines radius1, theta1, and phi1.
        :type mins: tuple(int, int, int)

        :arg  maxs: Defines radius2, theta2, and phi2.
        :type maxs: tuple(int, int, int)

        Returns: MO object
        """
        return self.subset(geom="rtp", **minus_self(locals()))

    def grid2grid(self, ioption: str, name: Optional[str] = None):
        """
        Convert a mesh with one element type to a mesh with another

        :arg ioption: type of conversion:
            quadtotri2   quad to 2 triangles, no new points.
            prismtotet3   prism to 3 tets, no new points.
            quadtotri4   quad to 4 triangles, with one new point.
            pyrtotet4   pyramid to 4 tets, with one new point.
            hextotet5   hex to 5 tets, no new points.
            hextotet6   hex to 6 tets, no new points.
            prismtotet14   prism to 14 tets, four new points (1 + 3 faces).
            prismtotet18   prism to 18 tets, six new points (1 + 5 faces).
            hextotet24   hex to 24 tets, seven new points (1 + 6 faces).
            tree_to_fe   quadtree or octree grid to grid with no parent-type elements.
        :type option: str
        :arg name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str

        Returns MO object
        """
        if name is None:
            name = make_name("mo", self._parent.mo.keys())
        cmd = "/".join(["grid2grid", ioption, name, self.name])
        self.sendcmd(cmd)
        self._parent.mo[name] = MO(name, self._parent)
        return self._parent.mo[name]

    def grid2grid_tree_to_fe(self, name: Optional[str] = None):
        """
        Quadtree or octree grid to grid with no parent-type elements.
        :arg name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str

        Returns MO object
        """
        return self.grid2grid(ioption="tree_to_fe", **minus_self(locals()))

    def grid2grid_quadtotri2(self, name: Optional[str] = None):
        """
        Quad to 2 triangles, no new points.
        :arg name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str

        Returns MO object
        """
        return self.grid2grid(ioption="quadtotri2", **minus_self(locals()))

    def grid2grid_prismtotet3(self, name: Optional[str] = None):
        """
        Quad to 2 triangles, no new points.
        Prism to 3 tets, no new points.
        :arg name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str

        Returns MO object
        """
        return self.grid2grid(ioption="prismtotet3", **minus_self(locals()))

    def grid2grid_quadtotri4(self, name: Optional[str] = None):
        """
        Quad to 4 triangles, with one new point
        :arg name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str

        Returns MO object
        """
        return self.grid2grid(ioption="quadtotri4", **minus_self(locals()))

    def grid2grid_pyrtotet4(self, name: Optional[str] = None):
        """
        Pyramid to 4 tets, with one new point
        :arg name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str

        Returns MO object
        """
        return self.grid2grid(ioption="pyrtotet4", **minus_self(locals()))

    def grid2grid_hextotet5(self, name: Optional[str] = None):
        """
        Hex to 5 tets, no new points
        :arg name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str

        Returns MO object
        """
        return self.grid2grid(ioption="hextotet5", **minus_self(locals()))

    def grid2grid_hextotet6(self, name: Optional[str] = None):
        """
        Hex to 6 tets, no new points
        :arg name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str

        Returns MO object
        """
        return self.grid2grid(ioption="hextotet6", **minus_self(locals()))

    def grid2grid_prismtotet14(self, name: Optional[str] = None):
        """
        Prism to 14 tets, four new points (1 + 3 faces)
        :arg name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str

        Returns MO object
        """
        return self.grid2grid(ioption="prismtotet14", **minus_self(locals()))

    def grid2grid_prismtotet18(self, name: Optional[str] = None):
        """
        Prism to 18 tets, four new points (1 + 3 faces)
        :arg name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str

        Returns MO object
        """
        return self.grid2grid(ioption="prismtotet18", **minus_self(locals()))

    def grid2grid_hextotet24(self, name: Optional[str] = None):
        """
        Hex to 24 tets, seven new points (1 + 6 faces)
        :arg name: Internal Lagrit name of new mesh object, automatically created if None
        :type name: str

        Returns MO object
        """
        return self.grid2grid(ioption="hextotet24", **minus_self(locals()))

    def connect(
        self,
        algorithm: Optional[str] = None,
        option: Optional[str] = None,
        stride: Optional[Tuple[int, int, int]] = None,
        big_tet_coords: List[Tuple[float, float, float]] = [],  # noqa: B006
    ):
        """
        Connect the nodes into a Delaunay tetrahedral or triangle grid.

        :arg algorithm: type of connect: delaunay, noadd, or check_interface
        :type algorithm: str
        :arg option: type of connect: noadd, or check_interface
        :type option: str
        :arg stride: tuple of (first, last, stride) of points
        :type stride: tuple(int)
        """
        cmd = ["connect"]
        if algorithm is not None:
            cmd.append(algorithm)
        if stride is not None and algorithm == "delaunay":
            cmd += [",".join(map(str, stride))]
            for b in big_tet_coords:
                cmd += [",".join(map(str, b))]
        if option is not None:
            cmd += [option]
        cmd = "/".join(cmd)
        self.sendcmd(cmd)

    def connect_delaunay(
        self,
        option: Optional[str] = None,
        stride: Optional[Tuple[int, int, int]] = None,
        big_tet_coords: List[Tuple[float, float, float]] = [],  # noqa: B006
    ):
        """
        Connect the nodes into a Delaunay tetrahedral or triangle grid without adding nodes.
        """
        mo_tmp = self.copypts()
        mo_tmp.setatt("imt", 1)
        mo_tmp.setatt("itp", 0)
        mo_tmp.rmpoint_compress(filter_bool=True)
        mo_tmp.connect(
            algorithm="delaunay",
            option=option,
            stride=stride,
            big_tet_coords=big_tet_coords,
        )
        self.sendcmd("/".join(["cmo", "move", self.name, mo_tmp.name]))

    def connect_noadd(self):
        """
        Connect the nodes into a Delaunay tetrahedral or triangle grid without adding nodes.
        """
        self.connect(algorithm="noadd")

    def connect_check_interface(self):
        """
        Connect the nodes into a Delaunay tetrahedral or triangle grid
        exhaustively checking that no edges of the mesh cross a material
        boundary.
        """
        self.connect(algorithm="check_interface")

    def copypts(self, elem_type="tet", name: Optional[str] = None):
        """
        Copy points from mesh object to new mesh object

        :arg name: Name to use within lagrit for the created mesh object
        :type name: str
        :arg mesh_type: Mesh type for new mesh
        :type mesh_type: str
        :returns: mesh object
        """
        if name is None:
            name = make_name("mo", self._parent.mo.keys())
        mo_new = self._parent.create(elem_type=elem_type, name=name)
        self.sendcmd("/".join(["copypts", mo_new.name, self.name]))
        return mo_new

    def extrude(
        self,
        offset: float,
        offset_type="const",
        return_type="volume",
        direction: Optional[Tuple[float, float, float]] = None,
        name: Optional[str] = None,
    ):
        """
        Extrude mesh object to new mesh object
        This command takes the current mesh object (topologically 1d or 2d mesh (a line, a set of line
        segments, or a planar or non-planar surface)) and extrudes it into three
        dimensions along either the normal to the curve or surface (default),
        along a user defined vector, or to a set of points that the user has specified.
        If the extrusion was along the normal of the surface or along a user
        defined vector, the command can optionally find the external surface of
        the volume created and return that to the user.
        Refer to http://lagrit.lanl.gov/docs/commands/extrude.html for more details on arguments.


        :arg name: Name to use within lagrit for the created mesh object
        :type name: str
        :arg offset: Distance to extrude
        :type offset: float
        :arg offset_type: either const or min (interp will be handled in the PSET class in the future)
        :type offset_type: str
        :arg return_type: either volume for entire mesh or bubble for just the external surface
        :type return_type: str
        :arg direction: Direction to extrude in, defaults to normal of the object
        :type direction: lst[float,float,float]
        :returns: mesh object
        """
        if name is None:
            name = make_name("mo", self._parent.mo.keys())
        cmd = ["extrude", name, self.name, offset_type, str(offset), return_type]
        if direction is not None:
            cmd.append(",".join(map(str, direction)))
        self.sendcmd("/".join(cmd))
        self._parent.mo[name] = MO(name, self._parent)
        return self._parent.mo[name]

    def refine_to_object(
        self,
        mo: "MO",
        level: Optional[int] = None,
        imt: Optional[int] = None,
        prd_choice: Optional[int] = None,
    ):
        """
        Refine mesh at locations that intersect another mesh object

        :arg mo: Mesh object to intersect with current mesh object to determine where to refine
        :type mo: PyLaGriT mesh object
        :arg level: max level of refinement
        :type level: int
        :arg imt: Value to assign to imt (LaGriT material type attribute)
        :type imt: int
        :arg prd_choice: directions of refinement
        :type prd_choice: int
        """

        itetlevbool = True
        if level == 1:
            itetlevbool = False
        if level is None:
            level = 1
            itetlevbool = False
        for _ in range(level):
            attr_name = self.intersect_elements(mo)
            if itetlevbool:
                e_attr = self.eltset_attribute(attr_name, 0, boolstr="gt")
                e_level = self.eltset_attribute("itetlev", level, boolstr="lt")
                e_refine = self.eltset_bool([e_attr, e_level], boolstr="inter")
                e_attr.delete()
                e_level.delete()
            else:
                e_refine = self.eltset_attribute(attr_name, 0, boolstr="gt")
            if prd_choice is not None:
                p_refine = e_refine.pset()
                p_refine.refine(prd_choice=prd_choice)
                p_refine.delete()
            else:
                e_refine.refine()
            e_refine.delete()
        if imt is not None:
            attr_name = self.intersect_elements(mo)
            e_attr = self.eltset_attribute(attr_name, 0, boolstr="gt")
            p = e_attr.pset()
            p.setatt("imt", 13)
            p.delete()

    def intersect_elements(self, mo: "MO", attr_name: Optional[str] = None):
        """
        This command takes two meshes and creates an element-based attribute in mesh1
        that contains the number of elements in mesh2 that intersected the respective
        element in mesh1. We define intersection as two elements sharing any common point.

        :arg mo: Mesh object to intersect with current mesh object to determine where to refine
        :type mo: PyLaGriT mesh object
        :arg attr_name: Name to give created attribute
        :type attr_name: str
        :returns: attr_name
        """
        attr_name = attr_name if attr_name else "attr00"
        self.sendcmd("/".join(["intersect_elements", self.name, mo.name, attr_name]))
        return attr_name

    def extract_surfmesh(
        self,
        name: Optional[str] = None,
        stride=(1, 0, 0),
        reorder=False,
        resetpts_itp=True,
        external=False,
    ):
        return self._parent.extract_surfmesh(
            name=name,
            cmo_in=self,
            stride=stride,
            reorder=reorder,
            resetpts_itp=resetpts_itp,
            external=external,
        )

    def interpolate(
        self,
        method: str,
        attsink: str,
        cmosrc: "MO",
        attsrc: str,
        stride=(1, 0, 0),
        tie_option: Optional[str] = None,
        flag_option: Optional[str] = None,
        keep_option: Optional[str] = None,
        interp_function: Optional[str] = None,
    ):
        """
        Interpolate values from attribute attsrc from mesh object cmosrc to current mesh object
        """
        stride = [str(v) for v in stride]
        cmd = [
            "interpolate",
            method,
            self.name,
            attsink,
            ",".join(stride),
            cmosrc.name,
            attsrc,
        ]
        if tie_option is not None:
            cmd += [tie_option]
        if flag_option is not None:
            cmd += [flag_option]
        if keep_option is not None:
            cmd += [keep_option]
        if interp_function is not None:
            cmd.append(interp_function)
        self.sendcmd("/".join(cmd))

    def interpolate_voronoi(
        self,
        attsink: str,
        cmosrc: "MO",
        attsrc: str,
        stride=(1, 0, 0),
        interp_function: Optional[str] = None,
    ):
        self.interpolate("voronoi", **minus_self(locals()))

    def interpolate_map(
        self,
        attsink: str,
        cmosrc: "MO",
        attsrc: str,
        stride=(1, 0, 0),
        tie_option: Optional[str] = None,
        flag_option: Optional[str] = None,
        keep_option: Optional[str] = None,
        interp_function: Optional[str] = None,
    ):
        self.interpolate("map", **minus_self(locals()))

    def interpolate_continuous(
        self,
        attsink: str,
        cmosrc: "MO",
        attsrc: str,
        stride=(1, 0, 0),
        interp_function: Optional[str] = None,
        nearest: Optional[str] = None,
    ):
        stride = [str(v) for v in stride]
        cmd = [
            "intrp",
            "continuous",
            self.name + " " + attsink,
            ",".join(stride),
            cmosrc.name + " " + attsrc,
        ]
        if nearest is not None:
            cmd += ["nearest", nearest]
        if interp_function is not None:
            cmd.append(interp_function)
        print("/".join(cmd))
        self.sendcmd("/".join(cmd))

    def interpolate_default(
        self,
        attsink: str,
        cmosrc: "MO",
        attsrc: str,
        stride=(1, 0, 0),
        tie_option="tiemax",
        flag_option="plus1",
        keep_option="delatt",
        interp_function: Optional[str] = None,
    ):
        self.interpolate("default", **minus_self(locals()))

    def copy(self, name: Optional[str] = None):
        """
        Copy mesh object
        """
        if name is None:
            name = make_name("mo", self._parent.mo.keys())
        self.sendcmd("/".join(["cmo/copy", name, self.name]))
        self._parent.mo[name] = MO(name, self._parent)
        return self._parent.mo[name]

    def stack_layers(
        self,
        filelist: List[str],
        file_type="avs",
        nlayers: Optional[List[int]] = None,  # ref_num
        matids: Optional[List[int]] = None,
        xy_subset: Optional[
            Tuple[float, float, float, float]
        ] = None,  # minx, miny, maxx, maxy
        buffer_opt: Optional[float] = None,
        truncate_opt: Optional[int] = None,
        pinchout_opt: Optional[float] = None,
        dpinchout_opt: Optional[Tuple[float, float]] = None,
        flip_opt=False,
    ):
        if nlayers is None:
            refine_num = [""] * (len(filelist) - 1)
        else:
            refine_num = [str(n) for n in nlayers]
        if matids is None:
            mat_num = ["1"] * len(filelist)
        else:
            mat_num = [str(m) for m in matids]
        cmd = ["stack/layers", file_type]
        if xy_subset is not None:
            cmd.append(",".join([str(v) for v in xy_subset]))

        cmd.append(" &")
        self.sendcmd("/".join(cmd), expectstr="\r\n")

        self._parent.sendcmd(
            " ".join([filelist[0], mat_num[0], "/ &"]), expectstr="\r\n"
        )
        for file, matid, n_refine in zip(
            filelist[1:-1], mat_num[1:-1], refine_num[0:-1]
        ):
            self._parent.sendcmd(
                " ".join([file, matid, n_refine, "/ &"]), expectstr="\r\n"
            )
        cmd = [" ".join([filelist[-1], mat_num[-1], refine_num[-1]])]

        if flip_opt is True:
            cmd.append("flip")
        if buffer_opt is not None:
            cmd.append(f"buffer {buffer_opt}")
        if truncate_opt is not None:
            cmd.append(f"trunc {truncate_opt}")
        if pinchout_opt is not None:
            cmd.append(f"pinch {pinchout_opt}")
        if dpinchout_opt is not None:
            cmd.append(f"dpinch {dpinchout_opt[0]}")
            cmd.append(f"dmin {dpinchout_opt[1]}")
        if not len(cmd) == 0:
            self._parent.sendcmd("/".join(cmd))

    def stack_fill(self, name: Optional[str] = None):
        if name is None:
            name = make_name("mo", self._parent.mo.keys())
        self.sendcmd("/".join(["stack/fill", name, self.name]))
        self._parent.mo[name] = MO(name, self._parent)
        return self._parent.mo[name]

    def math(
        self,
        operation: str,
        attsink: str,
        value: Optional[float] = None,
        stride: Tuple[int, int, int] = (1, 0, 0),
        cmosrc: Optional["MO"] = None,
        attsrc: Optional[str] = None,
    ):
        if cmosrc is None:
            cmosrc = self
        if attsrc is None:
            attsrc = attsink
        cmd = [
            "math",
            operation,
            self.name,
            attsink,
            ",".join([str(v) for v in stride]),
            cmosrc.name,
            attsrc,
        ]
        if value is not None:
            cmd += [str(value)]
        self.sendcmd("/".join(cmd))

    def settets(self, method: Optional[str] = None):
        if method is None:
            self.sendcmd("settets")
        else:
            self.sendcmd("settets/" + method)

    def settets_parents(self):
        self.settets("parents")

    def settets_geometry(self):
        self.settets("geometry")

    def settets_color_tets(self):
        self.settets("color_tets")

    def settets_color_points(self):
        self.settets("color_points")

    def settets_newtets(self):
        self.settets("newtets")

    def settets_normal(self):
        self.settets("normal")

    def triangulate(self, order="clockwise"):
        """
        triangulate will take an ordered set of nodes in the current 2d mesh object that define a perimeter of a polygon and create a trangulation of the polygon.  The nodes are assumed to lie in the xy plane; the z coordinate is ignored.  No checks are performed to verify that the nodes define a legal perimeter (i.e. that segments of the perimeter do not cross).  The code will connect the last node to the first node to complete the perimeter.

        This code support triangulation of self-intersecting polygons (polygon with holes), assuming that the order of the nodes are correct. Moreover the connectivity of the polyline must also be defined correctly. No checks are made.

        One disadvantage of the algorithm for triangulating self-intersecting polygons is that it does not always work. For example, if the holes have complicated shapes, with many concave vertices, the code might fail. In this case, the user may try to rotate the order of the nodes:
            NODE_ID:
                1 -> 2
                2 -> 3
                ...
                N -> 1

            :param order: direction of point ordering
            :type order: string

            Example:
                >>> from pylagrit import PyLaGriT
                >>> # Create pylagrit object
                >>> lg = PyLaGriT()
                >>> # Define polygon points in clockwise direction
                >>> # and create tri mesh object
                >>> coords = [[0.0, 0.0, 0.0],
                >>>           [0.0, 1000.0, 0.0],
                >>>           [2200.0, 200.0, 0.0],
                >>>           [2200.0, 0.0, 0.0]]
                >>> motri = lg.tri_mo_from_polyline(coords)
                >>> # Triangulate polygon
                >>> motri.triangulate()
                >>> motri.setatt("imt", 1)
                >>> motri.setatt("itetclr", 1)
                >>> # refine mesh with successively smaller edge length constraints
                >>> edge_length = [1000, 500, 250, 125, 75, 40, 20, 15]
                >>> for i,l in enumerate(edge_length):
                >>>     motri.resetpts_itp()
                >>>     motri.refine(refine_option='rivara',refine_type='edge',values=[l],inclusive_flag='inclusive')
                >>>     motri.smooth()
                >>>     motri.recon(0)
                >>> # provide additional smoothing after the last refine
                >>> for i in range(5):
                >>>     motri.smooth()
                >>>     motri.recon(0)
                >>> # create delaunay mesh and clean up
                >>> motri.tri_mesh_output_prep()
                >>> # dump fehm files
                >>> motri.dump_fehm("nk_mesh00")
                >>> # view results
                >>> motri.paraview()

        """
        self.sendcmd("triangulate/" + order)

    def refine(
        self,
        refine_option="constant",
        field=" ",
        interpolation=" ",
        refine_type="element",
        stride=(1, 0, 0),
        values=[1.0],  # noqa: B006
        inclusive_flag="exclusive",
        prd_choice: Optional[int] = None,
    ):
        cmd = [
            "refine",
            refine_option,
            field,
            interpolation,
            refine_type,
            " ".join([str(v) for v in stride]),
            "/".join([str(v) for v in values]),
            inclusive_flag,
        ]
        if prd_choice is not None:
            cmd.append(f"amr {prd_choice}")
        self._parent.sendcmd("/".join(cmd))

    def regnpts(
        self,
        geom: str,
        ray_points: Tuple[  # xyz
            Tuple[float, float, float],
            Tuple[float, float, float],
            Tuple[float, float, float],
        ]
        | Tuple[Tuple[float, float, float], Tuple[float, float, float]]  # rtz
        | Tuple[Tuple[float, float, float]],  # rpt
        region: "Region",
        ptdist: str,
        stride=(1, 0, 0),
        irratio=0,
        rrz=0,
        maxpenetr=None,
    ):
        if isinstance(stride, PSet):
            stride = "pset get " + stride.name
        else:
            stride = ",".join([str(v) for v in stride])
        end = str(irratio) + " " + str(rrz)
        if maxpenetr is not None:
            end = end + "/" + str(maxpenetr)
        ptdist = str(ptdist)
        if geom == "xyz":
            assert len(ray_points) == 3, "ray_points must contain three sets of points"  # noqa: S101
            pts = ""
            for p in ray_points:
                assert (  # noqa: S101
                    len(p) == 3
                ), "each entry in ray_points must contain 3 (x,y,z) values"
                pts += ",".join(list(map(str, p))) + "/"
        elif geom == "rtz":
            assert len(ray_points) == 2, "ray_points must contain two sets of points"  # noqa: S101
            pts = ""
            for p in ray_points:
                assert (  # noqa: S101
                    len(p) == 3
                ), "each entry in ray_points must contain 3 (x,y,z) values"
                pts += " &\n" + ",".join(list(map(str, p))) + "/"
        elif geom == "rtp":
            assert len(ray_points) == 2, "ray_points must contain one set of points"  # noqa: S101
            pts = ""
            for p in ray_points:
                assert (  # noqa: S101
                    len(p) == 3
                ), "each entry in ray_points must contain 3 (x,y,z) values"
                pts += " &\n" + ",".join(list(map(str, p))) + "/"
        else:
            print("Error: geom must be of type xyz rtz or rtp")
            return
        name = region.name
        cmd = "/".join(["regnpts", name, ptdist, stride, geom, pts])
        cmd += end
        print(cmd)
        self.sendcmd(cmd)

    def regnpts_xyz(
        self,
        ray_points,
        region,
        ptdist,
        stride=(1, 0, 0),
        irratio=0,
        rrz=0,
        maxpenetr=None,
    ):
        """
        Generates points in a region previously defined by the region command. The points are generated by shooting rays through a user specified set of points from a plane and finding the intersection of each ray with the surfaces that define the region.

        :arg ray_points: three points that define plane which rays emante from
        :type ray_points: 3-tuple of float 3-tuples
        :arg region: region to generate points within
        :type region: Region
        :arg ptdist: parameter that determines point distribution pattern
        :type ptdist: int float or str
        :arg stride: points to shoot rays through
        :type stride: int or PSet
        :arg irratio: parameter that determines point distribution pattern
        :type irratio: int
        :arg rrz: ratio zoning value
        :type rrz: int or float
        :arg maxpenetr: maximum distance along ray that points will be distributed
        :type maxpenetr: int or float

            example:
            >>> import numpy as np
            >>> from pylagrit import PyLaGriT
            >>> import sys
            >>>
            >>> lg = PyLaGriT()
            >>> p1 = (30.0, 0.0, 0.0)
            >>> p2 = (30.0, 1.0, 0.0)
            >>> p3 = (30.0, 1.0, 0.1)
            >>> pts = [p1, p2, p3]
            >>>
            >>> npts = (3, 3, 3)
            >>> mins = (0, 0, 0)
            >>> maxs = (10, 10, 10)
            >>> # mesh = lg.create()
            >>> mesh = lg.createpts_xyz(npts, mins, maxs, "hex", connect=False)
            >>> rayend = mesh.pset_geom_xyz(mins, maxs, ctr=(5, 5, 5))
            >>> mesh.rmpoint_compress(filter_bool=True)
            >>> eighth = mesh.surface_box(mins, (5, 5, 5))
            >>> boolstr2 = "gt " + eighth.name
            >>> reg2 = mesh.region(boolstr2)
            >>> mesh.regnpts_xyz(pts, reg2, 1000, stride=rayend)
            >>> mesh.dump("regn_test.gmv")
        """
        self.regnpts(geom="xyz", **minus_self(locals()))

    def regnpts_rtz(
        self,
        ray_points,
        region,
        ptdist,
        stride=(1, 0, 0),
        irratio=0,
        rrz=0,
        maxpenetr=None,
    ):
        """
        Generates points in a region previously defined by the region command. The points are generated by shooting rays through a user specified set of points from a line and finding the intersection of each ray with the surfaces that define the region.

        :arg ray_points: two points that define cylinder which rays emante from
        :type ray_points: 2-tuple of float 3-tuples
        :arg region: region to generate points within
        :type region: Region
        :arg ptdist: parameter that determines point distribution pattern
        :type ptdist: int float or str
        :arg stride: points to shoot rays through
        :type stride: int or PSet
        :arg irratio: parameter that determines point distribution pattern
        :type irratio: int
        :arg rrz: ratio zoning value
        :type rrz: int or float
        :arg maxpenetr: maximum distance along ray that points will be distributed
        :type maxpenetr: int or float
        """
        self.regnpts("rtz", **minus_self(locals()))

    def regnpts_rtp(
        self,
        ray_points,
        region,
        ptdist,
        stride=(1, 0, 0),
        irratio=0,
        rrz=0,
        maxpenetr=None,
    ):
        """
        Generates points in a region previously defined by the region command. The points are generated by shooting rays through a user specified set of points from an origin point and finding the intersection of each ray with the surfaces that define the region.

        :arg ray_points: single (x,y,z) point that defines center of spher which rays emante from
        :type ray_points: float 3-tuple
        :arg region: region to generate points within
        :type region: Region
        :arg ptdist: parameter that determines point distribution pattern
        :type ptdist: int float or str
        :arg stride: points to shoot rays through
        :type stride: int or PSet
        :arg irratio: parameter that determines point distribution pattern
        :type irratio: int
        :arg rrz: ratio zoning value
        :type rrz: int or float
        :arg maxpenetr: maximum distance along ray that points will be distributed
        :type maxpenetr: int or float
        """
        self.regnpts("rtp", **minus_self(locals()))

    def setpts(self, no_interface=False, closed_surfaces=False):
        """
        Set point types and imt material by calling surfset and regset routines.

        :arg ray_points: single (x,y,z) point that defines center of spher which rays emante from
        :type ray_points: float 3-tuple
        :arg region: region to generate points within
        :type region: Region
        :arg ptdist: parameter that determines point distribution pattern
        :type ptdist: int float or str
        :arg stride: points to shoot rays through
        :type stride: int or PSet
        :arg irratio: parameter that determines point distribution pattern
        :type irratio: int
        :arg rrz: ratio zoning value
        :type rrz: int or float
        :arg maxpenetr: maximum distance along ray that points will be distributed
        :type maxpenetr: int or float

            example:
            >>> import numpy as np
            >>> from pylagrit import PyLaGriT
            >>> import sys
            >>> lg = PyLaGriT()
            >>> mesh = lg.create()
            >>> mins = (0, 0, 0)
            >>> maxs = (5, 5, 5)
            >>> eighth = mesh.surface_box(mins, maxs)
            >>> boolstr1 = "le " + eighth.name
            >>> boolstr2 = "gt " + eighth.name
            >>> reg1 = mesh.region(boolstr1)
            >>> reg2 = mesh.region(boolstr2)
            >>> mreg1 = mesh.mregion(boolstr1)
            >>> mreg2 = mesh.mregion(boolstr2)
            >>> mesh.createpts_xyz((10, 10, 10), (0, 0, 0), (10, 10, 10), connect=False)
            >>> mesh.setpts()
            >>> mesh.connect()
            >>> mesh.dump("setpts_test.gmv")
        """

        cmd = "setpts"
        if no_interface and closed_surfaces:
            print("Error: no_interface and closed_surfaces are mutually exclusive")
            return
        if no_interface:
            cmd += "/no_interface"
        elif closed_surfaces:
            cmd += "/closed_surfaces/reflect"
        self.sendcmd(cmd)

    def smooth(self, *args, **kwargs):
        if "algorithm" not in kwargs:
            self.sendcmd("smooth")
        else:
            cmd = ["smooth", "position", kwargs["algorithm"]]
            for a in args:
                cmd.append(a)
            self.sendcmd("/".join(cmd))

    def recon(self, option="", damage="", checkaxy=False):
        cmd = ["recon", str(option), str(damage)]
        if checkaxy:
            cmd.append("checkaxy")
        self.sendcmd("/".join(cmd))

    def filter(
        self,
        stride=(1, 0, 0),
        tolerance: Optional[float] = None,
        boolean: Optional[Literal["min"] | Literal["max"]] = None,
        attribute: Optional[str] = None,
    ):
        stride = [str(v) for v in stride]
        cmd = ["filter", " ".join(stride)]
        if tolerance is not None:
            cmd.append(str(tolerance))
        if boolean is not None and attribute is not None:
            cmd.append(boolean)
            cmd.append(attribute)
        elif (boolean is None and attribute is not None) or (
            boolean is not None and attribute is None
        ):
            print("Error: Both boolean and attribute must be specified together")
            return
        self.sendcmd("/".join(cmd))

    def tri_mesh_output_prep(self):
        """
        Prepare tri mesh for output, remove dudded points,
        ensure delaunay volumes, etc.
        Combination of lagrit commands:
        filter/1 0 0
        rmpoint/compress
        recon/1
        resetpts/itp
        """
        self.filter()
        self.rmpoint_compress()
        self.recon("1")
        self.resetpts_itp()

    def surface(self, name: Optional[str] = None, ibtype="reflect"):
        if name is None:
            name = make_name("s", self.surfaces.keys())
        cmd = "/".join(["surface", name, ibtype, "sheet", self.name])
        self.sendcmd(cmd)
        self.surfaces[name] = Surface(name, self)
        return self.surfaces[name]

    def surface_box(
        self,
        mins: Tuple[float, float, float],
        maxs: Tuple[float, float, float],
        name: Optional[str] = None,
        ibtype="reflect",
    ):
        if name is None:
            name = make_name("s", self.surfaces.keys())
        cmd = "/".join(
            [
                "surface",
                name,
                ibtype,
                "box",
                ",".join([str(v) for v in mins]),
                ",".join([str(v) for v in maxs]),
            ]
        )
        self.sendcmd(cmd)
        self.surfaces[name] = Surface(name, self)
        return self.surfaces[name]

    def surface_cylinder(
        self,
        coord1: Tuple[float, float, float],
        coord2: Tuple[float, float, float],
        radius: float,
        name: Optional[str] = None,
        ibtype="reflect",
    ):
        if name is None:
            name = make_name("s", self.surfaces.keys())
        cmd = "/".join(
            [
                "surface",
                name,
                ibtype,
                "cylinder",
                ",".join([str(v) for v in coord1]),
                ",".join([str(v) for v in coord2]),
                str(radius),
            ]
        )
        self.sendcmd(cmd)
        self.surfaces[name] = Surface(name, self)
        return self.surfaces[name]

    def surface_plane(
        self,
        coord1: Tuple[float, float, float],
        coord2: Tuple[float, float, float],
        coord3: Tuple[float, float, float],
        name: Optional[str] = None,
        ibtype="reflect",
    ):
        if name is None:
            name = make_name("s", self.surfaces.keys())
        cmd = "/".join(
            [
                "surface",
                name,
                ibtype,
                "plane",
                " &\n" + ",".join([str(v) for v in coord1]),
                " &\n" + ",".join([str(v) for v in coord2]),
                " &\n" + ",".join([str(v) for v in coord3]),
            ]
        )
        self.sendcmd(cmd)
        self.surfaces[name] = Surface(name, self)
        return self.surfaces[name]

    def region_bool(self, bool, name=None):
        """
        This method is deprecated and will be replaced by the MO.region() method in future releases.

        """
        self.region(**minus_self(locals()))

    def region(
        self,
        boolstr: str,
        name: Optional[str] = None,
    ):
        """
        Create region using boolean string

        :param boolstr: String of boolean operations
        :type boolstr: str
        :param name: Internal lagrit name for mesh object
        :type name: string
        :returns: Region

        Example:
            >>> from pylagrit import PyLaGriT
            >>> import numpy
            >>> lg = PyLaGriT()
            >>> mesh = lg.create()
            >>> mins = (0, 0, 0)
            >>> maxs = (5, 5, 5)
            >>> eighth = mesh.surface_box(mins, maxs)
            >>> boolstr1 = "le " + eighth.name
            >>> boolstr2 = "gt " + eighth.name
            >>> reg1 = mesh.region(boolstr1)
            >>> reg2 = mesh.region(boolstr2)
            >>> mreg1 = mesh.mregion(boolstr1)
            >>> mreg2 = mesh.mregion(boolstr2)
            >>> mesh.createpts_brick_xyz((10, 10, 10), (0, 0, 0), (10, 10, 10))
            >>> mesh.rmregion(reg1)
            >>> mesh.dump("reg_test.gmv")
        """
        if name is None:
            name = make_name("r", self.regions.keys())
        cmd = "/".join(["region", name, boolstr])
        self.sendcmd(cmd)
        self.regions[name] = Region(name, self)
        return self.regions[name]

    def mregion(self, boolstr: str, name: Optional[str] = None):
        """
        Create mregion using boolean string

        :param boolstr: String of boolean operations
        :type boolstr: str
        :param name: Internal lagrit name for mesh object
        :type name: string
        :returns: MRegion
        """
        if name is None:
            name = make_name("mr", self.mregions.keys())
        cmd = "/".join(["mregion", name, boolstr])
        self.sendcmd(cmd)
        self.mregions[name] = MRegion(name, self)
        return self.mregions[name]

    def rmregion(
        self, region: "Region", rmpoints=True, filter_bool=False, resetpts_itp=True
    ):
        """
        Remove points that lie inside region

        :param region: name of region points will be removed from
        :type region: Region
        """
        name = region.name
        cmd = "/".join(["rmregion", name])
        self.sendcmd(cmd)
        if rmpoints:
            self.rmpoint_compress(filter_bool=filter_bool, resetpts_itp=resetpts_itp)

    def quality(self, *args, quality_type: Optional[str] = None, save_att=False):
        cmd = ["quality"]
        if quality_type is not None:
            cmd.append(quality_type)
            if save_att:
                cmd.append("y")
            for a in args:
                cmd.append(a)
        self.sendcmd("/".join(cmd))

    def quality_aspect(self, save_att=False):
        self.quality(quality_type="aspect", save_att=save_att)

    def quality_edge_ratio(self, save_att=False):
        self.quality(quality_type="edge_ratio", save_att=save_att)

    def quality_edge_min(self, save_att=False):
        self.quality(quality_type="edge_min", save_att=save_att)

    def quality_edge_max(self, save_att=False):
        self.quality(quality_type="edge_max", save_att=save_att)

    def quality_angle(self, value: float, boolean="gt", save_att=False):
        self.quality(boolean, str(value), quality_type="angle", save_att=save_att)

    def quality_pcc(self):
        self.quality(quality_type="pcc")

    def rmmat(self, material_number: int, option="", exclusive=False):
        """
        This routine is used to remove points that are of a specified material value
        (itetclr for elements or imt for nodes). Elements with the specified material
        value are flagged by setting the element material type negative. They are not
        removed from the mesh object.
        :param material_number: Number of material
        :type material_number: int
        :param option: {'','node','element','all'}, 'node' removes nodes with imt=material_number, 'element' removes elements with itetclr=material_number, 'all' or '' removes nodes and elements with material_number equal to imt and itetclr, respectively
        :type option: str
        :param exclusive: if True, removes everything except nodes with imt=material and removes everything except elements with itetclr= material number.
        :type exclusive: bool
        """
        cmd = ["rmmat", str(material_number), option]
        if exclusive:
            cmd.append("exclusive")
        self.sendcmd("/".join(cmd))

    def rmmat_element(self, material_number: int, exclusive=False):
        """
        This routine is used to remove elements that are of a specified material value
        (itetclr for elements). Elements with the specified material value are flagged
        by setting the element material type negative. They are not removed from the mesh
        object.
        :param material_number: Number of material
        :type material_number: int
        :param exclusive: if True, removes everything except elements with itetclr=material number.
        :type exclusive: bool
        """
        self.rmmat(material_number, option="element", exclusive=exclusive)

    def rmmat_node(self, material_number: int, exclusive=False):
        """
        This routine is used to remove points that are of a specified material value
        (imt).
        :param material_number: Number of material (imt)
        :type material_number: int
        :param exclusive: if True, removes everything except nodes with imt=material_number
        :type exclusive: bool
        """
        self.rmmat(material_number, option="node", exclusive=exclusive)


class Surface:
    """Surface class"""

    def __init__(self, name: str, parent: MO):
        self.name = name
        self._parent = parent

    def __repr__(self):
        return self.name

    def release(self):
        cmd = "surface/" + self.name + "/release"
        self._parent.sendcmd(cmd)
        del self._parent.surfaces[self.name]


class PSet:
    """Pset class"""

    def __init__(self, name: str, parent: PyLaGriT | MO):
        self.name = name
        self._parent = parent

    def __repr__(self):
        return str(self.name)

    def delete(self):
        cmd = "pset/" + self.name + "/delete"
        self._parent.sendcmd(cmd)
        del self._parent.pset[self.name]

    @property
    def xmin(self):
        self.minmax_xyz(verbose=False)
        strarr = decode_binary(cast(MO, self._parent)._parent.before).splitlines()
        return float(strarr[4].split()[1])

    @property
    def xmax(self):
        self.minmax_xyz(verbose=False)
        strarr = decode_binary(cast(MO, self._parent)._parent.before).splitlines()
        return float(strarr[4].split()[2])

    @property
    def xlength(self):
        self.minmax_xyz(verbose=False)
        strarr = decode_binary(cast(MO, self._parent)._parent.before).splitlines()
        return int(strarr[4].split()[4])

    @property
    def ymin(self):
        self.minmax_xyz(verbose=False)
        strarr = decode_binary(cast(MO, self._parent)._parent.before).splitlines()
        return float(strarr[5].split()[1])

    @property
    def ymax(self):
        self.minmax_xyz(verbose=False)
        strarr = decode_binary(cast(MO, self._parent)._parent.before).splitlines()
        return float(strarr[5].split()[2])

    @property
    def ylength(self):
        self.minmax_xyz(verbose=False)
        strarr = decode_binary(cast(MO, self._parent)._parent.before).splitlines()
        return int(strarr[5].split()[4])

    @property
    def zmin(self):
        self.minmax_xyz(verbose=False)
        strarr = decode_binary(cast(MO, self._parent)._parent.before).splitlines()
        return float(strarr[6].split()[1])

    @property
    def zmax(self):
        self.minmax_xyz(verbose=False)
        strarr = decode_binary(cast(MO, self._parent)._parent.before).splitlines()
        return float(strarr[6].split()[2])

    @property
    def zlength(self):
        self.minmax_xyz(verbose=False)
        strarr = decode_binary(cast(MO, self._parent)._parent.before).splitlines()
        return int(strarr[6].split()[4])

    def minmax_xyz(self, stride=(1, 0, 0), verbose=True):
        cmd = "/".join(
            [
                "cmo/printatt",
                self._parent.name,
                "-xyz-",
                "minmax",
                "pset,get," + self.name,
            ]
        )
        self._parent.sendcmd(cmd, verbose=verbose)

    def minmax(self, attname: Optional[str] = None, stride=(1, 0, 0)):
        cast(MO, self._parent).printatt(
            attname=attname, stride=stride, pset=self, ptype="minmax"
        )

    def list(self, attname=None, stride=(1, 0, 0)):
        cast(MO, self._parent).printatt(
            attname=attname, stride=stride, pset=self, ptype="list"
        )

    def setatt(self, attname: str, value: int | float | str):
        cmd = "/".join(
            [
                "cmo/setatt",
                self._parent.name,
                attname,
                "pset get " + self.name,
                str(value),
            ]
        )
        self._parent.sendcmd(cmd)

    def refine(
        self,
        refine_type="element",
        refine_option="constant",
        interpolation=" ",
        prange=(-1, 0, 0),
        field=" ",
        inclusive_flag="exclusive",
        prd_choice: Optional[int] = None,
    ):
        prange = [str(v) for v in prange]
        cmd = [
            "refine",
            refine_option,
            field,
            interpolation,
            refine_type,
            "pset get " + self.name,
            ",".join(prange),
            inclusive_flag,
        ]
        if prd_choice is not None:
            cmd.append("amr " + str(prd_choice))
        self._parent.sendcmd("/".join(cmd))

    def eltset(self, membership="inclusive", name: Optional[str] = None):
        """
        Create eltset from pset

        :arg membership: type of element membership, one of [inclusive,exclusive,face]
        :type membership: str
        :arg name: Name of element set to be used within LaGriT
        :type name: str
        :returns: PyLaGriT EltSet object
        """
        if name is None:
            name = make_name("e", cast(MO, self._parent).eltset.keys())
        cmd = ["eltset", name, membership, "pset", "get", self.name]
        self._parent.sendcmd("/".join(cmd))
        cast(MO, self._parent).eltset[name] = EltSet(name, self._parent)
        return cast(MO, self._parent).eltset[name]

    def expand(self, membership="inclusive"):
        """
        Add points surrounding pset to pset

        :arg membership: type of element membership, one of [inclusive,exclusive,face]
        :type membership: str
        """
        e = self.eltset(membership=membership)
        self._parent.sendcmd("pset/" + self.name + "/delete")
        self = e.pset(name=self.name)

    def interpolate(
        self,
        method: str,
        attsink: str,
        cmosrc: MO,
        attsrc: str,
        interp_function: Optional[str] = None,
    ):
        """
        Interpolate values from attribute attsrc from mesh object cmosrc to current mesh object
        """
        cast(MO, self._parent).interpolate(
            method=method,
            attsink=attsink,
            stride=["pset", "get", self.name],
            cmosrc=cmosrc,
            attsrc=attsrc,
            interp_function=interp_function,
        )

    def interpolate_voronoi(self, attsink, cmosrc, attsrc, interp_function=None):
        self.interpolate("voronoi", **minus_self(locals()))

    def interpolate_map(
        self,
        attsink,
        cmosrc,
        attsrc,
        tie_option="tiemax",
        flag_option="plus1",
        keep_option="delatt",
        interp_function=None,
    ):
        self.interpolate("map", **minus_self(locals()))

    def interpolate_continuous(
        self,
        attsink,
        cmosrc,
        attsrc,
        interp_function=None,
        nearest: Optional[str] = None,
    ):
        cmd = [
            "intrp",
            "continuous",
            self.name + " " + attsink,
            ",".join(["pset", "get", self.name]),
            cmosrc.name + " " + attsrc,
        ]
        if nearest is not None:
            cmd += ["nearest", nearest]
        if interp_function is not None:
            cmd.append(interp_function)
        self._parent.sendcmd("/".join(cmd))

    def interpolate_default(
        self,
        attsink,
        cmosrc,
        attsrc,
        tie_option="tiemax",
        flag_option="plus1",
        keep_option="delatt",
        interp_function=None,
    ):
        self.interpolate("default", **minus_self(locals()))

    def dump(self, filerootname: str, zonetype="zone"):
        """
        Dump zone file of pset nodes
        :arg filerootname: rootname of files to create, pset name will be added to name
        :type filerootname: string
        :arg zonetype: Type of zone file to dump, 'zone' or 'zonn'
        :tpye zonetype: string
        """
        cmd = ["pset", self.name, zonetype, filerootname + "_" + self.name, "ascii"]
        self._parent.sendcmd("/".join(cmd))

    def scale(
        self,
        scale_type="relative",
        scale_geom="xyz",
        scale_factor=(1, 1, 1),
        scale_center=(0, 0, 0),
    ):
        """
        Scale pset nodes by a relative or absolute amount
        :arg scale_type: Scaling type may be 'relative' or 'absolute'
        :type scale_type: string
        :arg scale_geom: May be one of the geometry types 'xyz' (Cartesian), 'rtz' (cylindrical), or 'rtp' (spherical)
        :type scale_geom: string
        :arg scale_factor: If scale_factor is relative, scaling factors are unitless multipliers. If absolute, scaling factors are constants added to existing coordinates.
        :type scale_factor: list
        :arg scale_center: Geometric center to scale from
        :type scale_center: list
        """

        scale_type = scale_type.lower()
        scale_geom = scale_geom.lower()

        if scale_geom not in ["xyz", "rtz", "rtp"]:
            print("ERROR: 'scale_geom' must be one of 'xyz', 'rtz', or 'rtp'")
            return

        if scale_type not in ["relative", "absolute"]:
            print("ERROR: 'scale_type' must be one of 'relative' or 'absolute'")
            return

        scale_factor = [str(v) for v in scale_factor]
        scale_center = [str(v) for v in scale_center]

        cmd = [
            "scale",
            ",".join(["pset", "get", self.name]),
            scale_type,
            scale_geom,
            ",".join(scale_factor),
            ",".join(scale_center),
        ]
        self._parent.sendcmd("/".join(cmd))

    def perturb(self, xfactor: float, yfactor: float, zfactor: float):
        """
        This command moves node coordinates in the following manner.

        Three pairs of random numbers between 0 and 1 are generated.
        These pairs refer to the x, y and z coordinates of the nodes respectively.
        The first random number of each pair is multiplied by the factor given in
        the command. The second random number is used to determine
        if the calculated offset is to be added or subtracted from the coordinate.
        """

        cmd = [
            "perturb",
            ",".join(["pset", "get", self.name]),
            str(xfactor),
            str(yfactor),
            str(zfactor),
        ]
        self._parent.sendcmd("/".join(cmd))

    def trans(self, xold: Tuple[float, float, float], xnew: Tuple[float, float, float]):
        """
        Translate points within a pset by the linear translation from (xold, yold, zold) to (xnew, ynew, znew)

        :arg xold: Tuple containing point (xold, yold, zold) to translate from
        :type xold: tuple
        :arg xnew: Tuple containing point (xnew, ynew, znew) to translate to
        :type xnew: tuple
        """

        cmd = [
            "trans",
            ",".join(["pset", "get", self.name]),
            ",".join([str(v) for v in xold]),
            ",".join([str(v) for v in xnew]),
        ]
        self._parent.sendcmd("/".join(cmd))

    def smooth(self, *args, **kwargs):
        if "algorithm" not in kwargs:
            algorithm = " "
        else:
            algorithm = kwargs["algorithm"]
        cmd = ["smooth", "position", algorithm, "pset get " + self.name]
        for a in args:
            cmd.append(a)
        self._parent.sendcmd("/".join(cmd))

    def pset_attribute(
        self, attribute: str, value: int, comparison="eq", name: Optional[str] = None
    ):
        """
        Define PSet from another PSet by attribute

        :kwarg attribute: Nodes defined by attribute ID.
        :type  attribute: str

        :kwarg value: attribute ID value.
        :type  value: integer

        :kwarg comparison: attribute comparison, default is eq.
        :type  comparison: can use default without specifiy anything, or list[lt|le|gt|ge|eq|ne]

        :kwarg name: The name to be assigned to the PSet created.
        :type  name: str

        Returns: PSet object

        Usage: newpset = oldpset.pset_attribute('attribute','value','comparison')
        """
        if name is None:
            name = make_name("p", self._parent.pset.keys())

        cmd = "/".join(
            [
                "pset",
                name,
                "attribute " + attribute,
                "pset,get," + self.name,
                " " + comparison + " " + str(value),
            ]
        )

        self._parent.sendcmd(cmd)
        self._parent.pset[name] = PSet(name, self._parent)
        return self._parent.pset[name]


class EltSet:
    """EltSet class"""

    def __init__(self, name: str, parent: MO):
        self.name = name
        self.faceset: Optional[FaceSet] = None
        self._parent = parent

    def __repr__(self):
        return str(self.name)

    def delete(self):
        cmd = "eltset/" + self.name + "/delete"
        self._parent.sendcmd(cmd)
        del self._parent.eltset[self.name]

    def create_faceset(self, filename: Optional[str] = None):
        if filename is None:
            filename = "faceset_" + self.name + ".avs"
        motmpnm = make_name("mo_tmp", self._parent._parent.mo.keys())
        self._parent._parent.sendcmd("/".join(["cmo/copy", motmpnm, self._parent.name]))
        self._parent._parent.sendcmd("/".join(["cmo/DELATT", motmpnm, "itetclr0"]))
        self._parent._parent.sendcmd("/".join(["cmo/DELATT", motmpnm, "itetclr1"]))
        self._parent._parent.sendcmd("/".join(["cmo/DELATT", motmpnm, "facecol"]))
        self._parent._parent.sendcmd("/".join(["cmo/DELATT", motmpnm, "idface0"]))
        self._parent._parent.sendcmd("/".join(["cmo/DELATT", motmpnm, "idelem0"]))
        self._parent._parent.sendcmd("eltset / eall / itetclr / ge / 0")
        self._parent._parent.sendcmd("eltset/edel/not eall " + self.name)
        self._parent._parent.sendcmd("rmpoint / element / eltset get edel")
        self._parent._parent.sendcmd("rmpoint / compress")
        self._parent._parent.sendcmd(
            "/".join(["dump / avs2", filename, motmpnm, "0 0 0 2"])
        )
        self._parent._parent.sendcmd("cmo / delete /" + motmpnm)
        self.faceset = FaceSet(filename, self)
        return self.faceset

    def minmax(self, attname: Optional[str] = None, stride=(1, 0, 0)):
        self._parent.printatt(
            attname=attname, stride=stride, eltset=self, ptype="minmax"
        )

    def list(self, attname: Optional[str] = None, stride=(1, 0, 0)):
        self._parent.printatt(attname=attname, stride=stride, eltset=self, ptype="list")

    def refine(self, amr=""):
        """
        Refine elements in the element set

        Example:
            >>> from pylagrit import PyLaGriT
            >>> import numpy
            >>> import sys
            >>>
            >>> df = 0.0005  # Fault half aperture
            >>> lr = 7  # Levels of refinement
            >>> nx = 4  # Number of base mesh blocks in x direction
            >>> nz = 20  # Number of base mesh blocks in z direction
            >>> d_base = df * 2 ** (lr + 1)  # Calculated dimension of base block
            >>> w = d_base * nx  # Calculated width of model
            >>> d = d_base * nz  # Calculated depth of model
            >>>
            >>> lg = PyLaGriT()
            >>>
            >>> # Create discrete fracture mesh
            >>> dxyz = numpy.array([d_base, d_base, 0.0])
            >>> mins = numpy.array([0.0, -d, 0.0])
            >>> maxs = numpy.array([w, 0, 0])
            >>> mqua = lg.createpts_dxyz(
            ...     dxyz, mins, maxs, "quad", hard_bound=("min", "max", "min"), connect=True
            ... )
            >>>
            >>> for i in range(lr):
            >>>     prefine = mqua.pset_geom_xyz(mins-0.1,(0.0001,0.1,0))
            >>>     erefine = prefine.eltset()
            >>>     erefine.refine()
            >>>     prefine.delete()
            >>>     erefine.delete()
            >>>
            >>> mtri = mqua.copypts("triplane")
            >>> mtri.connect()
            >>> # Make sure that not nodes are lost during connect
            >>> if 'The mesh is complete but could not include all points.' in lg.before:
            >>>     print 'Error: Lost some points during connect, not completing mesh and exiting workflow!'
            >>>     print ''
            >>>     sys.exit()
            >>> mtri.tri_mesh_output_prep()
            >>> mtri.reorder_nodes(cycle="xic yic zic")
            >>> pfault = mtri.pset_geom_xyz(mins - 0.1, (0.0001, 0.1, 0))
            >>> psource = mtri.pset_geom_xyz(mins - 0.1, mins + 0.0001)
            >>> mtri.setatt("imt", 1)
            >>> pfault.setatt("imt", 10)
            >>> psource.setatt("imt", 20)
            >>>
            >>> mtri.paraview(filename="discrete_fracture.inp")
        """
        cmd = "/".join(
            ["refine", "eltset", "eltset,get," + self.name, "amr " + str(amr)]
        )
        self._parent.sendcmd(cmd)

    def pset(self, name: Optional[str] = None):
        """
        Create a pset from the points in an element set
        :arg name: Name of point set to be used within LaGriT
        :type name: str
        :returns: PyLaGriT PSet object
        """
        if name is None:
            name = make_name("p", self._parent.pset.keys())
        cmd = "/".join(["pset", name, "eltset", self.name])
        self._parent.sendcmd(cmd)
        self._parent.pset[name] = PSet(name, self._parent)
        return self._parent.pset[name]

    def setatt(self, attname: str, value: int | float | str):
        cmd = "/".join(
            [
                "cmo/setatt",
                self._parent.name,
                attname,
                "eltset, get," + self.name,
                str(value),
            ]
        )
        self._parent.sendcmd(cmd)


class Region:
    """Region class"""

    def __init__(self, name: str, parent: MO):
        self.name = name
        self._parent = parent

    def __repr__(self):
        return str(self.name)

    def release(self):
        cmd = "region/" + self.name + "/release"
        self._parent.sendcmd(cmd)
        del self._parent.regions[self.name]


class MRegion:
    """Region class"""

    def __init__(self, name: str, parent: MO):
        self.name = name
        self._parent = parent

    def __repr__(self):
        return str(self.name)

    def release(self):
        cmd = "mregion/" + self.name + "/release"
        self._parent.sendcmd(cmd)
        del self._parent.mregions[self.name]


class FaceSet:
    """FaceSet class"""

    def __init__(self, filename, parent):
        self.filename = filename
        self._parent = parent

    def __repr__(self):
        return str(self.filename)
