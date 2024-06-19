from pylagrit import PyLaGriT


lg = PyLaGriT()
mhex = lg.read_fehm("fehm.grid")
# mhex.paraview()

mtet = lg.create(elem_type="tet")
mtet = mhex.copypts()  # type: ignore
mtet.connect()
mtet.paraview()
