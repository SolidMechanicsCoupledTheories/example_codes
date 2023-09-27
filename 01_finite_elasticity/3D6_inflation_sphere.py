"""
3D Code for  isothermal finite hyperelasticity

Inflation of a spherical shell

Basic units:
Length: mm
Mass: kg
Time:  s
Derived units:
Force: milliNewtons
Stress: kPa 

Eric Stewart and Lallit Anand   
ericstew@mit.edu and anand@mit.edu   

August 2023
"""

# Fenics-related packages
from dolfin import *
# Numerical array package
import numpy as np
# Plotting packages
import matplotlib.pyplot as plt
plt.close('all')
# Current time package
from datetime import datetime

# Set level of detail for log messages (integer)
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
"""
Import mesh from gmsh and identify the 3D domain and its relevant boundaries
"""
# Initialize an empty mesh object
mesh = Mesh()

# Thick-walled sphere dimensions
Ro = Constant(11)  # mm outer radius
Ri = Constant(10)  # mm inner radius


# Read the *.xdmf file data into mesh object
with XDMFFile("meshes/spherical_shell.xdmf") as infile:
    infile.read(mesh)

# Read the 2D subdomain data stored in the *.xdmf file
mvc2d = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("meshes/facet_spherical_shell.xdmf") as infile:
    infile.read(mvc2d, "name_to_read")
    
# Mesh facets
facets = cpp.mesh.MeshFunctionSizet(mesh, mvc2d)

# Surface numbering from gmsh:
# Physical Surface("yBot", 12)  
# //+
# Physical Surface("zBot", 13) 
# //+
# Physical Surface("xBot", 14) 
# //+
#Physical Surface("inner_surf", 15)

# Extract initial mesh coords
x = SpatialCoordinate(mesh)
# Define surface area measure
ds = Measure("ds", domain=mesh, subdomain_data= facets)

'''''''''''''''''''''
MATERIAL PARAMETERS
Arruda-Boyce Model
'''''''''''''''''''''
Gshear_0 = Constant(280.0)            # Ground state shear modulus
lambdaL  = Constant(5.12)             # Locking stretch
Kbulk    = Constant(1000.0*Gshear_0)  # Bulk modulus

# Simulation time control-related params
t        = 0.0       # start time (s)
rampRate = 1.0e-1    # s^{-1}
Ttot     = 1.0/rampRate  # time for pressure rise to press_max (s) 
numSteps = 100
dt       = Ttot/numSteps # (fixed) step size

# Maximum  internal pressure
press_max = 35 # kPa

# Expression for the time-dependent traction BC
pressRamp = Expression("press_max*t/tRamp", \
                      press_max=press_max, t= 0.0, tRamp = Ttot, degree=1)

"""
Function spaces
"""
# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For  pressure
                                                 
#
TH = MixedElement([U2, P1])     # Taylor-Hood style mixed element
ME = FunctionSpace(mesh, TH)    # Total space for all DOFs

# Define actual functions with the required DOFs
w    = Function(ME)
u, p = split(w)  # displacement u, pressure p

# A copy of functions to store values in the previous step
w_old         = Function(ME)
u_old,  p_old = split(w_old)   

# Define test functions
w_test         = TestFunction(ME)                
u_test, p_test = split(w_test)   

# Define trial functions needed for automatic differentiation
dw = TrialFunction(ME)                  

#-------------------------------------------------------
# Subroutines for kinematics and constitutive equations 
#-------------------------------------------------------
# Deformation gradient 
def F_calc(u):
    Id = Identity(3) 
    F  = Id + grad(u)
    return F

def lambdaBar_calc(u):
    F = F_calc(u)
    C = F.T*F
    Cdis = J**(-2/3)*C
    I1 = tr(Cdis)
    lambdaBar = sqrt(I1/3.0)
    return lambdaBar

def zeta_calc(u):
    lambdaBar = lambdaBar_calc(u)
    # Use Pade approximation of Langevin inverse
    z    = lambdaBar/lambdaL
    z    = conditional(gt(z,0.95), 0.95, z) # Keep simulation from blowing up
    beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
    zeta = (lambdaL/(3*lambdaBar))*beta
    return zeta

# Generalized shear modulus for Arruda-Boyce model
def Gshear_AB_calc(u):
    zeta    = zeta_calc(u)
    Gshear  = Gshear_0 * zeta
    return Gshear

#---------------------------------------------
# Subroutine for calculating the Cauchy stress
#---------------------------------------------
def T_calc(u,p):
    Id = Identity(3) 
    F   = F_calc(u)
    J = det(F)
    B = F*F.T
    Bdis = J**(-2/3)*B
    Gshear  = Gshear_AB_calc(u)
    T = (1/J)* Gshear * dev(Bdis) - p * Id
    return T

#----------------------------------------------
# Subroutine for calculating the Piola  stress
#----------------------------------------------
def Tmat_calc(u, p):
    Id = Identity(3) 
    F   = F_calc(u)
    J = det(F)
    #
    T   = T_calc(u,p)
    #
    Tmat   = J * T * inv(F.T)
    return Tmat

#----------------------------------------------
# Evaluate kinematics and constitutive relations
#----------------------------------------------
#
F =  F_calc(u)  
J = det(F)
lambdaBar = lambdaBar_calc(u)

# Piola stress
Tmat = Tmat_calc(u, p)

''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of forces (test fxn: u)
# Res_1: Coupling pressure (test fxn: p)

# Surface numbering from gmsh:
# Physical Surface("yBot", 12)  
# //+
# Physical Surface("zBot", 13) 
# //+
# Physical Surface("xBot", 14) 
# //+
#Physical Surface("inner_surf", 15)

# Normal vector to facets
n = FacetNormal(ds) 

# Cofactor of F
Fcof = J*inv(F.T)

# Configuration-dependent traction
traction = - pressRamp*dot(Fcof,n)

# The weak form for the balance of forces
Res_0 =  inner(Tmat, grad(u_test) )*dx   - dot(traction, u_test)*ds(15) 

# The weak form for the pressure
fac_p = ln(J)/J
#
Res_1 = dot( (p/Kbulk + fac_p), p_test)*dx
        
# Total weak form
Res = Res_0 +  Res_1 

# Automatic differentiation tangent:
a = derivative(Res, w, dw)

'''''''''''''''''''''
 SET UP OUTPUT FILES
'''''''''''''''''''''
# Set up output file for visualizationvisualization
#
file_results = XDMFFile("results/3D_sphere_inflate.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True


# Function space for projection of results
W2 = FunctionSpace(mesh, U2) # Vector space for visulization  
W = FunctionSpace(mesh,P1)   # Scalar space for visulization 

def writeResults(t):
    
       # Variable projecting and renaming
        u_Vis = project(u, W2)
        u_Vis.rename("disp"," ")
        
        # Visualize the pressure
        p_Vis = project(p, W)
        p_Vis.rename("p"," ")
        
        # Visualize J
        J_Vis = project(J, W)
        J_Vis.rename("J"," ")    
    
        # Visualize effective stretch
        lambdaBar_Vis = project(lambdaBar,W)
        lambdaBar_Vis.rename("LambdaBar"," ")
        
        P11_Vis = project(Tmat[0,0],W)
        P11_Vis.rename("P11, kPa","")
        P22_Vis = project(Tmat[1,1],W)
        P22_Vis.rename("P22, kPa","")    
        P33_Vis = project(Tmat[2,2],W)
        P33_Vis.rename("P33, kPa","")        
        
        # Visualize the Mises stress  
        T    = Tmat*F.T/J
        T0   = T - (1/3)*tr(T)*Identity(3)
        #
        Mises = sqrt((3/2)*inner(T0, T0))
        Mises_Vis = project(Mises,W)
        Mises_Vis.rename("Mises, kPa"," ")    

       # Write field quantities of interest
        file_results.write(u_Vis, t)
        file_results.write(p_Vis, t)
        file_results.write(J_Vis, t)  
        file_results.write(lambdaBar_Vis, t)
        #
        file_results.write(P11_Vis, t)  
        file_results.write(P22_Vis, t)    
        file_results.write(P33_Vis, t)  
        file_results.write(Mises_Vis, t)              
        
# Write initial state to XDMF file
writeResults(t=0.0)    

"""
Start simulation
"""
print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time 
startTime = datetime.now()

"""""""""""""""""
     STEP 
"""""""""""""""""
# Give the step a descriptive name
step = "Pressurize"

'''''''''''''''''''''''
DISPLACEMENT BOUNDARY CONDITIONS
'''''''''''''''''''''''      
# Surface numbering from gmsh:
# Physical Surface("yBot", 12)  
# //+
# Physical Surface("zBot", 13) 
# //+
# Physical Surface("xBot", 14) 
# //+
#Physical Surface("inner_surf", 15)


bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 14)  # u1 fix - xBot

bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 12)  # u2 fix - yBot

bcs_3 = DirichletBC(ME.sub(0).sub(2), 0, facets, 13)  # u3 fix - zBot


# BC set
bcs = [bcs_1, bcs_2, bcs_3]

'''''''''''''''''''''''
Define  the nonlinear variational problem
'''''''''''''''''''''''
ElasProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(ElasProblem)
#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-8
prm['newton_solver']['maximum_iterations'] = 30

#
# Initialize arrays for storing results
totSteps  = numSteps+1
timeHist0 = np.zeros(shape=[totSteps])
timeHist1 = np.zeros(shape=[totSteps]) 
timeHist2 = np.zeros(shape=[totSteps]) 

#Iinitialize a counter for reporting data
ii=0

# Time-stepping solution procedure loop
while (round(t + dt, 9) <= Ttot):

    # increment time
    t += dt
    # increment counter
    ii += 1
    
    # Update time variable in time-dependent BCs 
    pressRamp.t = t
    
    # Solve the problem
    try:
        (iter, converged) = solver.solve()
    except: # Break the loop if solver fails
        break
    
    # Write output to *.xdmf file
    writeResults(t)
    
    # Update DOFs for next step
    w_old.vector()[:] = w.vector()
 

    # Store  displacement and pressure at a particular point  at this time
    timeHist0[ii] = t 
    timeHist1[ii] = min(press_max*t/Ttot, press_max)  # time history of applied pressure
    timeHist2[ii] = w.sub(0).sub(0)(Ri, 0, 0)         # time history of displacement of 
                                                      # selected point

   # Print progress of calculation
    if ii%1 == 0:      
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: {} | Simulation Time: {} s, Wallclock Time: {}".\
              format(step, round(t,4), current_time))
        print("Iterations: {}".format(iter))
        print()  
             
# End analysis
print("-----------------------------------------")
print("End computation")                 
# Report elapsed real time for the analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("------------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("------------------------------------------")

'''''''''''''''''''''
    VISUALIZATION
'''''''''''''''''''''

# set plot font to size 14
font = {'size'   : 14}
plt.rc('font', **font)

# Get array of default plot colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Only plot as far as we have time history data
ind = np.argmax(timeHist0)

# Pressure versus displacement curve:  
#
fig = plt.figure() 
#fig.set_size_inches(7,4)
ax=fig.gca()  
plt.plot(timeHist1[:ind], timeHist2[:ind], c='b', linewidth=2.0, marker='.')
#-------------------------------------------------------------
#ax.set.xlim(-0.01,0.01)
#ax.set.ylim(-0.03,0.03)
#plt.axis('tight')
plt.grid(linestyle="--", linewidth=0.5, color='b')
ax.set_xlabel(r'Internal pressure, kPa')              
ax.set_ylabel(r'Displacement of inner wall, mm')
ax.set_title("Internal pressure versus radial expansion curve", size=14, weight='normal')
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.show()

fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/pressurise_sphere.png", dpi=600)
