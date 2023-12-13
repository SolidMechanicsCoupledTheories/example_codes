"""
Code for large deformation plane strain rate-independent elasto-plasticity
#
Units:
#---------------------------------
Length: mm
Mass  : 1000 kg = 1 tonne
Time  :  s
#---------------------------------
Force       : N
Stress      : MPa 
Energy .....: mJ
Mass density: tonne/mm^3 # 1 tonne = 1000 kg
#
 Lallit Anand   and    Eric Stewart
anand@mit.edu        ericstew@mit.edu

November 2023

"""
# Fenics-related packages
from dolfin import *
import numpy as np
from sympy import Abs
#from mshr import *
import math
# Plotting packages
import matplotlib.pyplot as plt
plt.close('all') # clear all plots from previous siGshearlation.
# Current time package
from datetime import datetime  

# Set level of detail for log messages (integer)
# 
# Guide:
# CRITICAL  = 50, // errors that may lead to data corruption
# ERROR     = 40, // things that HAVE gone wrong
# WARNING   = 30, // things that MAY go wrong later
# INFO      = 20, // information of general interest (includes solver info)
# PROGRESS  = 16, // what's happening (broadly)
# TRACE     = 13, // what's happening (in detail)
# DBG       = 10  // sundry
#
set_log_level(30)

# The behavior of the form compiler FFC can be adjusted by prescribing
# various parameters. Here, we want to use the UFLACS backend of FFC::
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 4

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''
#
# Initialize an empty mesh object
mesh = Mesh()

# Read the .xdmf  file data into mesh object
with XDMFFile("meshes/plate_holes.xdmf") as infile:
    infile.read(mesh)
    
# Extract initial mesh coords
x = SpatialCoordinate(mesh)
    
# Read mesh boundary facets.
# Specify parameters of 1-D mesh function object
mvc_1d = MeshValueCollection("size_t", mesh, 1)
# Read the   *.xdmf  data into mesh function object
with XDMFFile("meshes/facet_plate_holes.xdmf") as infile:
    infile.read(mvc_1d, "name_to_read")
    
# Store boundary facets data
facets = cpp.mesh.MeshFunctionSizet(mesh, mvc_1d)

# Gmsh labels
# //+
# Physical Curve("left", 15) 
# //+
# Physical Curve("right", 16)  

# Facet normal
n2D = FacetNormal(mesh)
n   = as_vector([n2D[0], n2D[1], 0.0])
# Area measure on facets
ds = Measure('ds',domain = mesh, subdomain_data = facets)  

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
# Elastic prperties
Eyoung  = Constant(100.E3)          # GPa
nu      = Constant(0.3)
Gshear  = Eyoung/(2*(1+nu))  
Kbulk   = Eyoung/(3*(1-nu))

# Plastic propertie; Voce strain--hardening
Y_init  = Constant(250.e0)  
Y_sat   = Constant(500.e0)
H_init  = Constant(5000.e0)

"""
Time parameters
"""
# Displacement amplitude
uMax = 0.5      
# Initialize time
t = 0
# Time to reach uMax
Ttot = 1.0   
dt = 0.01

# Time step field, constant within body
dk = Constant(dt)

#
dispRamp = Expression(("uMax*(t/Tramp)"),
                  uMax = uMax, Tramp = Ttot, t = 0.0, degree=1)
                      
"""
Function spaces
"""
#
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # vector displacement  
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # scalar equiv. tensile plastic strain 
T1 = TensorElement("Lagrange", mesh.ufl_cell(), 1) # tensor plastic defgrad

ME = FunctionSpace(mesh, U2) 

u      = Function(ME)
u_old  = Function(ME)
u_test = TestFunction(ME) 

# Define trial functions needed for automatic differentiation
du = TrialFunction(ME)

# Initialize  Fp
# 
T1_state   = FunctionSpace(mesh, T1) 
Fp2d_old   = Function(T1_state)
#
# Assign  2d identity as initial value for Fp2d
Fp2d_init =  project(Identity(2), T1_state) 
# Assign Fp_init as th old value of Fp
Fp2d_old.assign(Fp2d_init)

# Initialize ebarP
P1_state = FunctionSpace(mesh, P1) 
ebarP_old   = Function(P1_state)
#
# Assign zero as initial value for ebarP
ebarP_init =  project(Constant(0.0), P1_state) 
# Assign ebarP_init as th old value of ebarP
ebarP_old.assign(ebarP_init)

"""
Subroutines
"""
# Special gradient operators for plane strain functions 
#
# Gradient of vector field u   
def grad_vector(u):
    grad_u = grad(u)
    return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, 0]]) 

# Gradient of scalar field y
# (just need an extra zero for dimensions to work out)
def grad_scalar(y):
    grad_y = grad(y)
    return as_vector([grad_y[0], grad_y[1], 0.])

#  3d Deformation gradient 
def F_calc(u):
    dim = len(u)
    Id = Identity(dim)          # Identity tensor
    F = Id + grad(u)            # 2D Deformation gradient
    F = as_tensor([[F[0,0], F[0,1], 0],
                  [F[1,0], F[1,1], 0],
                  [0, 0, 1]]) # Full F
    return F

#  Full 3d Fp
def Fp_calc(Fp2d):  
    Fp = as_tensor([[Fp2d[0,0], Fp2d[0,1], 0],
                  [Fp2d[1,0], Fp2d[1,1], 0],
                  [0, 0, 1]])   
    return Fp

# Subroutine for elastic second Piola stress Te
def Te_calc(u, Fp2d):
    
    F     =  F_calc(u)
    
    Fp    =  Fp_calc(Fp2d)
    
    Fe    =  F*inv(Fp)
    
    Ee    =  0.5*(Fe.T*Fe - Identity(3))
    
    Ee0   =  Ee - 1/3*tr(Ee)* Identity(3)
    
    Te    = 2*Gshear*Ee0  + Kbulk * tr(Ee)* Identity(3)
    
    return Te


# Subroutine for evaluating direction of plastic flow  Np
def Np_calc(u, Fp2d_old):
    
    Fp_old =  Fp_calc(Fp2d_old)
    
    Te_tr  = Te_calc(u, Fp2d_old)
    
    Te0_tr = Te_tr - 1/3*tr(Te_tr)* Identity(3)
    
    sigbar_tr = sqrt(3./2.*inner(Te0_tr, Te0_tr))
    
    Np     =  sqrt(3./2.)*Te0_tr/sigbar_tr
    
    return conditional( gt(sigbar_tr,1.e-4) , Np, Identity(3) )

# Subroutine for calculating debarP
def DebarP_calc(u, Fp2d_old, ebarP_old):
    #
    Fp_old =  Fp_calc(Fp2d_old)
    
    Te_tr  = Te_calc(u,Fp2d_old)

    Te0_tr = Te_tr - 1/3 * tr(Te_tr) * Identity(3)
    
    sigbar_tr = sqrt(3./2.*inner(Te0_tr, Te0_tr)) 
    
    Y_n    = Y_sat - (Y_sat - Y_init)*exp(-(H_init/Y_sat)*ebarP_old)
    
    f_tr   = sigbar_tr - Y_n     
    
    H_n    = H_init*(1- Y_n/Y_sat)
    
    DebarP = (sigbar_tr - Y_n)/(3*Gshear + H_n)
        
    return conditional( gt(f_tr,0), DebarP, 0.)

# Subroutine for updating Fp
def Fp_update(u,  Fp2d_old, ebarP_old):
    
    Debarp = DebarP_calc(u,Fp2d_old, ebarP_old)
    
    Np     = Np_calc(u, Fp2d_old)
    
    Fp_old  = Fp_calc(Fp2d_old)
    
    Fp_new = Fp_old + sqrt(3/2)*Debarp * Np * Fp_old
    
    # Normalize Fp so that its determiant is unity
    
    Fp_new = det(Fp_new)**(-1/3)*Fp_new 
    
    return Fp_new

#  Subroutine for updating 2d Fp
def Fp2d_update(Fp):  
    Fp2d_new = as_tensor([ [Fp[0,0], Fp[0,1]],
                           [Fp[1,0], Fp[1,1]]   ])
    return Fp2d_new

def ebarP_update(u, Fp2d_old, ebarP_old):
    
    Debarp = DebarP_calc(u, Fp2d_old, ebarP_old)
    
    ebarP_new  = ebarP_old + DebarP

    return ebarP_new


# Subroutine for  Piola stress
def Piola_calc(u, Fp2d):
    F     =  F_calc(u)
    
    Fp    =  Fp_calc(Fp2d)
    
    Fe    =  F*inv(Fp)
    
    Te = Te_calc(u, Fp2d)
 
    Piola = Fe*Te*inv(Fp.T)
    
    return Piola

"""
Kinematics and constitutive equations
"""
# Kinematics
F  = F_calc(u)
J  = det(F)


# New 3d Fp
Fp = Fp_update(u, Fp2d_old, ebarP_old)
Jp = det(Fp)

# New 2d Fp
Fp2d = Fp2d_update(Fp)

#
Fe = F*inv(Fp)
Je = det(Fe)
Ee = 0.5*(Fe.T*Fe-Identity(3))

#  Evaulate the  Piola stress
Piola = Piola_calc(u, Fp2d)

#  Evaluate DebarP
DebarP = DebarP_calc(u, Fp2d_old, ebarP_old)

#   ebarP update
ebarP  = ebarP_update(u, Fp2d_old, ebarP_old)

#-----------------------------------------------------------------------
"""
Weak form:
"""
# The weak form for the equilibrium equation
# 
Res = inner(Piola, grad_vector(u_test) )*dx 

# Automatic differentiationtangent:
a = derivative(Res, u, du)

'''''''''''''''''''''
SET UP OUTPUT FILES
'''''''''''''''''''''
# Output file setup
file_results = XDMFFile("results/2D_plate_with_holes.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

#  Define vector and scalar spaces for  visualization of results
W2 = FunctionSpace(mesh, U2)   # Vector space  
W  = FunctionSpace(mesh, P1)   # Scalar space 

# Subroutine for writing results  to XDMF at time t
def writeResults(t):
    
    # Project the displacement for visualiztion
    u_Vis = project(u, W2)
    u_Vis.rename("disp"," ")

    # Project ebarP for visualiztion
    ebarP_Vis = project(ebarP, W)
    ebarP_Vis.rename("peeq"," ")    
    
    # Project Jp for visualiztion
    Jp_Vis = project(Jp, W)
    Jp_Vis.rename("Jp"," ")   

    # Visualize the Mises stress  
    F    = F_calc(u)
    J    = det(F)
    T    = Piola*F.T/J
    T0   = T - (1/3)*tr(T)*Identity(3)
    p    = - (1/3)*tr(T)
    #
    Mises = sqrt((3/2)*inner(T0, T0))
    Mises_Vis = project(Mises,W)
    Mises_Vis.rename("Mises"," ")   
    p_Vis = project(p, W)
    p_Vis.rename("p"," ")  
    
    # Project  Piola stress for visualization
    T11_Vis = project(T[0,0],W)
    T11_Vis.rename("T11"," ")
    
    T22_Vis = project(T[1,1],W)
    T22_Vis.rename("T22"," ")  
    
    T33_Vis = project(T[2,2],W)
    T33_Vis.rename("T33"," ")      
    
    T21_Vis = project(T[1,0],W)
    T21_Vis.rename("T21"," ")      
    
    T31_Vis = project(T[2,0],W)
    T31_Vis.rename("T31"," ")    
    
    
   # Write field quantities of interest
    file_results.write(u_Vis, t)   
    file_results.write(T11_Vis, t)
    file_results.write(T22_Vis, t) 
    file_results.write(T33_Vis, t) 
    file_results.write(T21_Vis, t) 
    file_results.write(T31_Vis, t)     
    file_results.write(Mises_Vis, t)  
    file_results.write(p_Vis, t) 
    file_results.write(ebarP_Vis, t)  
    file_results.write(Jp_Vis, t) 

# Write initial state to XDMF file
writeResults(t=0)

print("------------------------------------")
print("Start Simulation")
print("------------------------------------")

# Store start time 
startTime = datetime.now()

"""""""""""""""""
     STEP
"""""""""""""""""
# Give the step a descriptive name
step = "Tension"

'''''''''''''''''''''''
BOUNDARY CONDITIONS
'''''''''''''''''''''''    
# Gmsh labels
# //+
# Physical Curve("left", 15) 
# //+
# Physical Curve("right", 16)  


# Dirichlet boundary conditions
bc1 = DirichletBC(ME.sub(0), Constant(0.0), facets, 15)  #  u1 fix  left
bc2 = DirichletBC(ME.sub(1), Constant(0.0), facets, 15)  #  u2 fix  left


bc3 = DirichletBC(ME.sub(0), dispRamp, facets, 16)  #  u1 move right

bcs = [bc1, bc2, bc3]

#-----------------------------------------------------------------------
# Set up the non-linear solver
StressProblem = NonlinearVariationalProblem(Res, u, bcs, J=a)
solver  = NonlinearVariationalSolver(StressProblem)

#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" #'lu'   #'petsc'   #'gmres'
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-7
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['relaxation_parameter'] = 1.0
prm['newton_solver']['error_on_nonconvergence'] = False # Set this to False for adaptive time stepping
                                                        # Else set it to True
#  Create files to store resuts
#
siz  = 10000 
timeHist0 = np.zeros([siz])  
timeHist1 = np.zeros([siz])  
timeHist2 = np.zeros([siz])  
# Set counter to store results
ii = 0

while  round(t, 2) < round(Ttot,2):
    
    t += dt
   
    # Update time variable in the time-dependent BC
    dispRamp.t = t 
       
    # Solve the problem
    (iter, converged) = solver.solve()
    
    # Check if solver converges
    if converged: 
        
        # Print progress of calculation periodically
        if ii%1 == 0:      
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
            print("Simulation Time: {} s | dt: {} s".format(round(t,4), round(dt, 4)))
            print()   
        
        # Iteration-based adaptive time-stepping
        if iter<=3:
            dt = 1.5*dt
            dk.assign(dt)
        elif iter>=20:
            dt = dt/1.5
            dk.assign(dt)
        elif dt> 0.01:
            dt =0.01
            dk.assign(dt)
            
        # Write output to *.xdmf file
        writeResults(t)

        # Update state variables. 
        # First, project the state variable fields to the nodes.  
        ebarP_proj =  project(ebarP, P1_state)
        Fp2d_proj  =  project(Fp2d, T1_state)
        #
        # Then, update "old" state variables for the next step.
        ebarP_old.assign( ebarP_proj )
        Fp2d_old.assign( Fp2d_proj )
        #
        # Finally, update DOFs for next step
        u_old.vector()[:] = u.vector()   
        
        # Increment counter
        ii += 1
        
        # Store the current time
        timeHist0[ii] = t 
    
        # Calculate traction
        #
        Piola    = Piola_calc(u,Fp2d)
        traction = dot(Piola,n)
        # Calculate reaction force 
        Force    = dot(traction,n)*ds(16)    
        # Store the results for displacement and Force 
        timeHist1[ii]   = uMax*(t/Ttot)      
        timeHist2[ii]   = assemble(Force)  # Convert from UFL operator to a number    

    # If solver doesn't converge, do not save results and try a smaller dt 
    else: 
        # back up in time
        t = t - float(dk)
        
        # cut back on dt
        dt = dt/2
        dk.assign(dt)
        
        # Reset DOFs for next step
        u.vector()[:] = u_old.vector()
        
        # Reset Fp2d_old and ebarP_old to old values
        Fp2d_old.assign(project(Fp2d_old, T1_state))
        ebarP_old.assign(project(ebarP_old, P1_state))


# End analysis
print("-----------------------------------------")
print("End computation")                 
# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("------------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("------------------------------------------")    
    


'''''''''''''''''''''
    VISUALIZATION
'''''''''''''''''''''
# Post processing of results:    
#
# Set plot font to size 14
font = {'size'   : 14}
plt.rc('font', **font)

# Get array of default plot colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors     = prop_cycle.by_key()['color']

# Only plot as far as we have time history data
ind = np.argmax(timeHist0) +1
#  Force versus axial dispalcement curve:  
#
fig = plt.figure() 
#fig.set_size_inches(7,4)
ax=fig.gca()  
plt.plot(timeHist1[:ind], timeHist2[:ind]/1.E3, c='b', linewidth=2.0, marker='.')
#-------------------------------------------------------------
#ax.set.xlim(-0.01,0.01)
#ax.set.ylim(-0.03,0.03)
#plt.axis('tight')
plt.grid(linestyle="--", linewidth=0.5, color='b')
ax.set_xlabel("Displacement, mm",size=14)
ax.set_ylabel("Force, kN",size=14)
ax.set_title("Axial force versus displacement", size=14, weight='normal')
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
#ax.yaxis.set_minor_formatter(FormatStrFormatter("%.32f"))
import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
plt.show()    

fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/2d_pe_plate_with_holes.png", dpi=600)
   