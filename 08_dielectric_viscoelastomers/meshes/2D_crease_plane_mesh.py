"""
2D plane with imperfection mesh, for electro-creasing studies.

"""
"""
Step 1. Create a mesh with pygmsh 
"""
import pygmsh


# Geometry parameters
l = 6 # mm
imperf = 1e-3 # as a fraction of l

# Resolution parameters
res_crse = l/10 # coarse element size
res_fine = l/1000 # fine element size


# Create an empty geometry, and initialize it
geometry = pygmsh.geo.Geometry()
# Create  a model to add data to
model = geometry.__enter__()

# Add points 
points = [model.add_point((0, 0), mesh_size=res_crse),   #0
          model.add_point((0, l*(1-imperf) ), mesh_size=res_fine),       # 1
          model.add_point((l*imperf, l ), mesh_size=res_fine),       # 2
          model.add_point((l/6, l), mesh_size=res_crse),       # 3
          model.add_point((l/6, 0), mesh_size=res_crse)]       # 4

# Add lines 
line1 = model.add_line(points[0], points[1])
line2 = model.add_line(points[1], points[2])
line3 = model.add_line(points[2], points[3])
line4 = model.add_line(points[3], points[4])
line5 = model.add_line(points[4], points[0])

# Create a line_loop and plane_surface for meshing
lines_loop = model.add_curve_loop([line1, line2, line3, line4, line5])
plane_surface = model.add_plane_surface(lines_loop)

# Call gmsh before adding physical entities
model.synchronize()

# The final step before mesh generation is to mark the domain  
# and the different boundaries. Give these entities names so that 
# they can be identified in gmsh 
model.add_physical(line2, "Imperfection")
model.add_physical(line3, "Top")
model.add_physical(line5, "Bottom")
model.add_physical(line1, "Left")
model.add_physical(line4, "Right")
model.add_physical([plane_surface], "Area")

"""
In Fenics and Paraview these geometrical entities are numbered as:
   Imperfection = 1
   Top          = 2
   Bottom       = 3
   Left         = 4
   Right        = 5
   Area         = 6
"""

"""
Step 2.  Write the mesh to file using gmsh
"""
import gmsh
geometry.generate_mesh(dim=2)
gmsh.write("meshes/pygmsh_plane.msh")
gmsh.clear()
geometry.__exit__()

# Now that we have saved the mesh to a `msh` file, we would like
# to convert it to a format that interfaces with dolfin/fenics. 

"""
Step 3. convert the mesh to.xdmf  format using meshio
"""
import meshio
mesh_from_file = meshio.read("meshes/pygmsh_plane.msh")


"""
Step. 4  Extract cells and boundary data.

Now that we have created the mesh, we need to extract the cells 
and physical data. We need to create a separate file for the 
facets (lines),  which we will use when we define boundary 
conditions in  Fenics. We do this  with the following convenience 
function. Note that as we would like a  2 dimensional mesh, we need to 
remove the z-values in the mesh coordinates, if any.
"""
import numpy
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells},\
                           cell_data={"name_to_read":[cell_data]})
    return out_mesh

"""
Step 5.
With this function in hand, we can save the facet line mesh 
and the domain triangle  mesh in `XDMF` format 
"""

line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write("meshes/facet_pygmsh_plane.xdmf", line_mesh)

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write("meshes/pygmsh_plane.xdmf", triangle_mesh)


