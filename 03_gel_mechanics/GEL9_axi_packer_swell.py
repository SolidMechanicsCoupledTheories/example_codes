"""
Code for hydrogels

Degrees of freedom:
#
Displacement: u
pressure: p
chemical potential: mu
concentration: c
    
Units:
#
Length: mm
Mass: kg
Time: s
Mass density: kg/mm^3
Force: milliN
Stress: kPa 
Energy: microJ
Temperature: K
Amount of substance: mol
Species concentration: mol/mm^3
Chemical potential: milliJ/mol
Molar volume: mm^3/mol
Species diffusivity: mm^2/s
Gas constant: microJ/(mol K)

Eric Stewart and Lallit Anand   
ericstew@mit.edu and anand@mit.edu   

August 2023  
"""

# Fenics-related packages
from dolfin import *
from ufl import sign
# Numerical array package
import numpy as np
# Plotting packages
import matplotlib.pyplot as plt
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

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 4      

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''
# Create mesh 
r_in   = 3 # inner packer radius
r_out  = 9 # outer packer radius
r_wall = 11 # outer rigid wall radius
L0     = 6 # axial length of packer

# Last two numbers below are the number of elements in the two directions
mesh = RectangleMesh(Point(r_in, 0), Point(r_out, L0), 10, 10, "crossed")
x = SpatialCoordinate(mesh)

# Identify the boundary entities of mesh
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],r_in) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],r_out) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],L0) and on_boundary 
    
# Identify the bottom left corner of the domain which will be fixed  
def Ground(x, on_boundary):
        return near(x[0],r_in) and near(x[1], L0/2)

# Mark boundary subdomains
facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
# First, mark all boundaries with common index
DomainBoundary().mark(facets, 5) 
# Next mark specific boundaries
Left().mark(facets,   1)
Bottom().mark(facets, 2)
Right().mark(facets,  3)
Top().mark(facets,    4)
 
# Define the boundary integration measure "ds".
ds = Measure('ds', domain=mesh, subdomain_data=facets)

# Facet normal
n = as_vector([FacetNormal(mesh)[0], FacetNormal(mesh)[1], 0])

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''

Gshear_0= Constant(300.0)         # Shear modulus, kPa
lambdaL = Constant(2.5)            # Locking stretch. 
Kbulk   = Constant(1000*Gshear_0)  # Bulk modulus, kPa
Omega   = Constant(1.00e5)         # Molar volume of fluid
D       = Constant(5.00e-2)        # Diffusivity
chi     = Constant(0.6)            # Flory-Huggins mixing parameter
theta0  = Constant(298)            # Reference temperature
R_gas   = Constant(8.3145e6)       # Gas constant
RT      = 8.3145e6*theta0 
#
phi0    = Constant(0.999)          # Initial polymer volume fraction
mu0     = ln(1.0-phi0) + phi0 + chi*phi0*phi0  #Initialize chemical potential

"""
Simulation time-control related params
"""

t    = 0.0          # initialization of time
Ttot = 3600*6       # total simulation time 
ttd  = 400.0       # Decay time constant
dt   = 200       # Fixed step size

# Boundary condition expression for increasing  the chemical potential
# 
muAmp = Expression(("mu0*exp(-t/td)"),
                mu0 = float(mu0), td = ttd, t = 0.0, degree=1)

'''''''''''''''''''''
FEM SETUP
'''''''''''''''''''''

# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For  normalized chemical potential and  normalized species concentration
#
TH = MixedElement([U2, P1, P1, P1]) # Taylor-Hood style mixed element
ME = FunctionSpace(mesh, TH)    # Total space for all DOFs

# Define actual functions with the required DOFs
w = Function(ME)
u, p, mu, c = split(w)  # displacement u, chemical potential  mu,  concentration c

# A copy of functions to store values in the previous step for time-stepping
w_old = Function(ME)
u_old,  p_old,  mu_old, c_old = split(w_old)   

# Define test functions in 
w_test = TestFunction(ME)                
u_test, p_test, mu_test, c_test = split(w_test)   

#Define trial functions neede for automatic differentiation
dw = TrialFunction(ME)             

# Initialize chemical potential, corresponding to nearly dry polymer.
mu0 = ln(1.0-phi0) + phi0 + chi*phi0*phi0
init_mu = Constant(mu0) 
mu_init = interpolate(init_mu,ME.sub(2).collapse())
assign(w_old.sub(2),mu_init)
assign(w.sub(2), mu_init)

# Assign initial  species normalized concentration c0
c0 = 1/phi0 - 1
init_c = Constant(c0)
c_init = interpolate(init_c, ME.sub(3).collapse())
assign(w_old.sub(3),c_init)
assign(w.sub(3), c_init)

'''''''''''''''''''''
Subroutines
'''''''''''''''''''''
# Special gradient operators for Axisymmetric test functions 
#
# Gradient of vector field u   
def axi_grad_vector(u):
    grad_u = grad(u)
    return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, u[0]/x[0]]]) 

# Gradient of scalar field y
# (just need an extra zero for dimensions to work out)
def axi_grad_scalar(y):
    grad_y = grad(y)
    return as_vector([grad_y[0], grad_y[1], 0.])

# Axisymmetric deformation gradient 
def F_axi_calc(u):
    dim = len(u)
    Id = Identity(dim)          # Identity tensor
    F = Id + grad(u)            # 2D Deformation gradient
    F33_exp =  (x[0]+u[0])/x[0]  # axisymmetric F33, R/R0 
    F33 = conditional(lt(x[0], DOLFIN_EPS), 1.0, F33_exp) # avoid divide by zero at r=0  
    return as_tensor([[F[0,0], F[0,1], 0],
                  [F[1,0], F[1,1], 0],
                  [0, 0, F33]]) # Full axisymmetric F

def lambdaBar_calc(u):
    F = F_axi_calc(u)
    C = F.T*F
    I1 = tr(C)
    lambdaBar = sqrt(I1/3.0)
    return lambdaBar

def zeta_calc(u):
    lambdaBar = lambdaBar_calc(u)
    # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
    # This is sixth-order accurate.
    z    = lambdaBar/lambdaL
    z    = conditional(gt(z,0.95), 0.95, z) # Keep from blowing up
    beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
    zeta = (lambdaL/(3*lambdaBar))*beta
    return zeta

def zeta0_calc():
    # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
    # This is sixth-order accurate.
    z    = 1/lambdaL
    z    = conditional(gt(z,0.95), 0.95, z) # Keep from blowing up
    beta0 = z*(3.0 - z**2.0)/(1.0 - z**2.0)
    zeta0 = (lambdaL/3)*beta0
    return zeta0

#  Elastic Je
def Je_calc(u,c):
    F = F_axi_calc(u)  
    detF = det(F)   
    #
    detFs = 1.0 + c          # = Js
    Je    = (detF/detFs)     # = Je
    return   Je    

# Normalized Piola stress for Arruda_Boyce material
def Piola_calc(u,p):
    F     = F_axi_calc(u)
    zeta  = zeta_calc(u)
    zeta0 = zeta0_calc()
    Tmat = (zeta*F - zeta0*inv(F.T) ) - J*p*inv(F.T)/Gshear_0
    return Tmat

# Normalized species flux
def Flux_calc(u, mu, c):
    F = F_axi_calc(u) 
    #
    Cinv = inv(F.T*F) 
    #
    Mob = (D*c)/(Omega*RT)*Cinv
    #
    Jmat = - RT* Mob * axi_grad_scalar(mu)
    return Jmat


# Macaulay bracket function
def ppos(x):
    return (x+abs(x))/2.


'''''''''''''''''''''''''''''
Kinematics and constitutive relations
'''''''''''''''''''''''''''''
# Kinematics
F = F_axi_calc(u)
J = det(F)  # Total volumetric jacobian

# Elastic volumetric Jacobian
Je     = Je_calc(u,c)                    
Je_old = Je_calc(u_old,c_old)

#  Normalized Piola stress
Tmat = Piola_calc(u, p)

#  Normalized species  flux
Jmat = Flux_calc(u, mu, c)

'''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''

# Residuals:
# Res_0: Balance of forces (test fxn: u)
# Res_1: Pressure variable (test fxn: p)
# Res_2: Balance of mass   (test fxn: mu)
# Res_3: Auxiliary variable (test fxn: c)
# Res_Contact: contact force for packer wall

# Time step field, constant  
dk = Constant(dt)

# The weak form for the equilibrium equation
Res_0 = inner(Tmat, axi_grad_vector(u_test) )*x[0]*dx

# Penalty weak form for the contact at the x[0] = r_wall surface
#
# The mathematical form of the penalty force is a simple linear relationship,
#
# f = - k * < \delta u >,
#
# where f is a horizontal contact force,
#       k is a penalty stiffness parameter, and
#       \delta u is the amount by which the material surface coordinates 
#                penetrate the contact plane.
#       < x > is the Macaulay bracket, which returns: 0 if x < 0
#                                                     x if x > 0
#             (so that f=0 when contact isn't occurring.)
#
l_pen       = r_wall # x-coord of contact plane
k_pen       = 1.0e4 # penalty stiffness, mN/mm
f_pen       = -k_pen*ppos(  x[0] + u[0] - l_pen ) # spatial penalty force, scalar
Res_contact = -dot( f_pen, u_test[0] )*x[0]*ds(3)  # weak form contribution
#
# # This block is my attempt to implement friction, but I haven't gotten it to work.
# from ufl import sign
# fric_coeff = 0.0
# fric_dir   = sign(u[1] - u_old[1])
# fric_pen   = fric_coeff*f_pen*fric_dir
# Res_fric   = dot(fric_pen, u_test[1])*x[0]*ds(3)

# The weak form for the auxiliary pressure variable definition
Res_1 = dot((p*Je/Kbulk + ln(Je)) , p_test)*x[0]*dx

# The weak form for the mass balance of solvent      
Res_2 = dot((c - c_old)/dk, mu_test)*x[0]*dx \
        -  Omega*dot(Jmat , axi_grad_scalar(mu_test) )*x[0]*dx

# The weak form for the concentration
fac = 1/(1+c)
fac1 =  mu - ( ln(1.0-fac)+ fac + chi*fac*fac)
fac2 = - (Omega*Je/RT)*p 
fac3 = fac1 + fac2 
#
Res_3 = dot(fac3, c_test)*x[0]*dx
        
# Total weak form
Res = Res_0 + Res_1 + Res_2 + Res_3 + Res_contact


# Automatic differentiation tangent:
a = derivative(Res, w, dw)
   
'''''''''''''''''''''''
BOUNDARY CONDITIONS
'''''''''''''''''''''''      

# Boundary condition definitions
bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 1)  # u1 fix - left  
bcs_2 = DirichletBC(ME.sub(0), Constant((0, 0)), Ground, method='pointwise') # Fix the bottom left corner
bcs_3 = DirichletBC(ME.sub(2), muAmp, facets, 2)         # chem. pot. - bottom
bcs_4 = DirichletBC(ME.sub(2), muAmp, facets, 3)         # chem. pot. - right
bcs_5 = DirichletBC(ME.sub(2), muAmp, facets, 4)         # chem. pot. - top

# BCs set 
bcs = [bcs_1, bcs_2, bcs_3, bcs_4, bcs_5]

'''''''''''''''''''''
Define nonlinear problem
'''''''''''''''''''''
GelProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(GelProblem)

#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-8
prm['newton_solver']['maximum_iterations'] = 30

'''''''''''''''''''''
Set-up output files
'''''''''''''''''''''
# Output file setup
file_results = XDMFFile("results/axi_packer_swell.xdmf")
# "Flush_output" permits reading the output during simulation
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Function space for projection of results
W2 = FunctionSpace(mesh, U2)   # Vector space for visualization  
W  = FunctionSpace(mesh,P1)    # Scalar space for visualization 
   
def writeResults(t):
      # Variable projecting and renaming
      u_Vis = project(u, W2)
      u_Vis.rename("disp"," ")
      
      # Visualize the pressure
      p_Vis = project(p, W)
      p_Vis.rename("p"," ")
      
      # Visualize  the normalized chemical potential
      mu_Vis = project(mu, W)
      mu_Vis.rename("mu"," ")
      
      # Visualize  the normalized concentration
      c_Vis = project(c, W)
      c_Vis.rename("c"," ")
      
      # Visualize phi
      phi = 1/(1+c)
      phi_Vis = project(phi, W)
      phi_Vis.rename("phi"," ")
      
      # Visualize J
      J_Vis = project(J, W)
      J_Vis.rename("J"," ")    
      
      # Visualize effective stretch
      lambdaBar = lambdaBar_calc(u)
      lambdaBar_Vis = project(lambdaBar,W)
      lambdaBar_Vis.rename("LambdaBar"," ")
      
      # Visualize Je
      Je_Vis = project(Je, W)
      Je_Vis.rename("Je"," ")    
      
      # Visualize some components of Piola stress
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
      file_results.write(mu_Vis, t)
      file_results.write(c_Vis, t)
      file_results.write(phi_Vis, t)
      #
      file_results.write(J_Vis, t)  
      file_results.write(lambdaBar_Vis, t)
      file_results.write(Je_Vis, t)  
      #
      file_results.write(P11_Vis, t)  
      file_results.write(P22_Vis, t)    
      file_results.write(P33_Vis, t)  
      file_results.write(Mises_Vis, t)            
     
    
# Write initial values
writeResults(t=0.0)

# initalize output array and counter for time history variables
timeHist0 = np.zeros(shape=[10000])
timeHist1 = np.zeros(shape=[10000])
iii = 0


print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time 
startTime = datetime.now()

# Give the step a descriptive name
step = "Swell"

while (t < Ttot):
    
    # increment time
    t += dt
    
    # increment counter 
    iii += 1

    # update time variables in time-dependent BCs
    muAmp.t = t

    # Solve the problem
    (iter, converged) = solver.solve()  
    
    # Update DOFs for next step
    w_old.vector()[:] = w.vector()
    
    # Write output to *.xdmf file
    writeResults(t)
    
    # record time history variables
    timeHist0[iii] = t
    timeHist1[iii] = assemble(dot(Gshear_0*Tmat*n, n)*x[0]*ds(3))
    
    # Print progress of calculation periodically
    if t%20 == 0:      
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: {} |   Simulation Time: {}  s  |  Wallclock Time: {}".format(step, round(t,9), current_time))
        print("Iterations: {}".format(iter))
        print()      
        
# End analysis
print("------------------------------------------")
print("End computation")                 
# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("-------------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("-------------------------------------------")
 
'''''''''''''''''''''
    VISUALIZATION
'''''''''''''''''''''

# set plot font to size 14
font = {'size'   : 14}
plt.rc('font', **font)

# Only plot as far as we have time history data
ind = np.argmax(timeHist0)

# Get array of default plot colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure()
plt.plot(timeHist0[0:ind]/3600, -timeHist1[0:ind]/1e3, linewidth=2.0,\
         color=colors[2])
plt.axvline(0.6, c='k', linewidth=1.5, linestyle = "--", label="Contact")
plt.axis('tight')
plt.ylabel(r"Normal Force (N)")
plt.xlabel(r"Time (h)")
plt.grid(linestyle="--", linewidth=0.5, color='b')
plt.ylim(0,22)
plt.xlim(0,6)
plt.legend()

fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/packer_swell_force.png", dpi=600)