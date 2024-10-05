from pylagrit import PyLaGriT


lg = PyLaGriT()
m = lg.create()
m.createpts_xyz(
    (3, 3, 3), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), rz_switch=[1, 1, 1], connect=True
)
m.status()
m.status(brief=True)
fs = m.create_boundary_facesets(base_name="faceset_bounds")
m.dump_exo("cube.exo", facesets=fs.values())
