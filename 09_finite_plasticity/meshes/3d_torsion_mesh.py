"""
3d_torsion_mesh.py
#
This program is used to create a mesh for  torsion of
bar with a square cross-section.
 
Units [L]: mm, [F]: N, [Stress]: MPa
"""
#
import pygmsh
#

geom = pygmsh.occ.Geometry()

model3D = geom.__enter__()

# Use this line to generate the cylinder mesh
bar =  model3D.add_cylinder(x0=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 10.0], radius=5.0, mesh_size=2.0)

# Use this line to generate the box mesh
#bar =  model3D.add_box([0.0, 0.0, 0.0], [10.0, 10.0, 20],mesh_size=1.0)


model3D.synchronize()


model3D.add_physical(bar,"Bar")

geom.generate_mesh()

import gmsh
gmsh.write("mesh3D.msh")
model3D.__exit__()

import numpy
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells},\
                           cell_data={"name_to_read":[cell_data]})
    return out_mesh

import meshio
mesh3D_from_msh = meshio.read("mesh3D.msh")
tetra_mesh = create_mesh(mesh3D_from_msh, "tetra")
meshio.write("meshes/3d_torsion_mesh.xdmf", tetra_mesh)


