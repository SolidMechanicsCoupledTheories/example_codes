"""
Code for Gels

3D s buckling of a thin cylinder

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

"""
Import mesh
"""
# Thin corona parameters
re = 50    # external radius
ri = 30    # internal radius
h0 = 1  # thickness

# Initialize an empty mesh object
mesh = Mesh()

# Read the .xdmf  file data into mesh object
with XDMFFile("meshes/thin_corona.xdmf") as infile:
    infile.read(mesh)
    
# Read the 2D subdomain data stored in the *.xdmf file
mvc2d = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("meshes/facet_thin_corona.xdmf") as infile:
    infile.read(mvc2d, "name_to_read")
    
# Mesh facets
facets = cpp.mesh.MeshFunctionSizet(mesh, mvc2d)

# Surface labels form Gmsh
#
# Physical Surface("right_bot", 29)
# //+
# Physical Surface("left_top", 30)
# //+
# Physical Surface("inner_surf", 31)
# //+
# Physical Surface("z_bot", 32)
# //+
# Physical Surface("z_top", 33)

# Extract initial mesh coords
x = SpatialCoordinate(mesh)
# Define surface area measure
ds = Measure("ds", domain=mesh, subdomain_data= facets)

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
# Set the locking stretch to a large number to model a Neo-Hookean material
# A low value  of the locking stretch helps in stablizing the buckling  solution
#
Gshear_0= Constant(1000.0)         # Shear modulus, kPa
lambdaL = Constant(100)            # Locking stretch. 
Kbulk   = Constant(1000*Gshear_0)  # Bulk modulus, kPa
Omega   = Constant(1.00e5)         # Molar volume of fluid
D       = Constant(5.00e-3)        # Diffusivity
chi     = Constant(0.1)            # Flory-Huggins mixing parameter
theta0  = Constant(298)            # Reference temperature
R_gas   = Constant(8.3145e6)       # Gas constant
RT      = 8.3145e6*theta0 
#
phi0    = Constant(0.999)          # Initial polymer volume fraction
mu0     = ln(1.0-phi0) + phi0 + chi*phi0*phi0  #Initialize chemical potential

"""
Simulation time-control related params
"""
t    = 0.0     # initialization of time
Ttot = 1000     # total simulation time 
ttd  = 300   # Decay time constant
dt   = 25       # Fixed step size

# Boundary condition expressions as necessary
muAmp = Expression(("mu0*exp(-t/td)"),
                mu0 = mu0, td = ttd, t = 0.0, degree=1)

'''''''''''''''''''''
Function spaces
'''''''''''''''''''''
# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For  normalized chemical potential and  normalized species concentration
#
TH = MixedElement([U2, P1, P1, P1]) # Taylor-Hood style mixed element
ME = FunctionSpace(mesh, TH)    # Total space for all DOFs

# Define trial functions
w = Function(ME)
u, p, mu, c = split(w)  # displacement u, chemical potential  mu,  concentration c

# A copy of functions to store values in the previous step
w_old = Function(ME)
u_old,   p_old,  mu_old,  c_old  = split(w_old)   

# Define test functions
w_test = TestFunction(ME)                
u_test, p_test, mu_test, c_test  = split(w_test)   

# Define trial functions needed for automatic differentiation
dw = TrialFunction(ME)             

# Assign initial value of normalized chemical potential
init_mu = Constant(mu0) 
mu_init = interpolate(init_mu,ME.sub(2).collapse())
assign(w_old.sub(2),mu_init)
assign(w.sub(2), mu_init)

# Assign initial  value of  normalized concentration
c0 = 1/phi0 - 1
#
init_c = Constant(c0)
c_init = interpolate(init_c, ME.sub(3).collapse())
assign(w_old.sub(3), c_init)
assign(w.sub(3), c_init)

'''''''''''''''''''''
SUBROUTINES
'''''''''''''''''''''
# Deformation gradient 
def F_calc(u):
    Id = Identity(3) 
    F  = Id + grad(u)
    return F

def lambdaBar_calc(u):
    F = F_calc(u)
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
    F = F_calc(u)  
    detF = det(F)   
    #
    detFs = 1.0 + c          # = Js
    Je    = (detF/detFs)     # = Je
    return   Je    

# Normalized Piola stress for Arruda_Boyce material
def Piola_calc(u,p):
    F     = F_calc(u)
    zeta  = zeta_calc(u)
    zeta0 = zeta0_calc()
    Tmat  = (zeta*F - zeta0*inv(F.T) ) - J*p*inv(F.T)/Gshear_0
    return Tmat

# Normalized species flux
def Flux_calc(u, mu, c):
    F = F_calc(u) 
    #
    Cinv = inv(F.T*F) 
    #
    Mob = (D*c)/(Omega*RT)*Cinv
    #
    Jmat = - RT* Mob * grad(mu)
    return Jmat

'''''''''''''''''''''''''''''
Kinematics and constitutive relations
'''''''''''''''''''''''''''''
# Kinematics
F = F_calc(u)
J = det(F)  # Total volumetric jacobian

# Elastic volumetric Jacobian
Je     = Je_calc(u,c)                    
Je_old = Je_calc(u_old,c_old)

#  Normalized Piola stress
Tmat = Piola_calc(u, p)

#  Normalized species  flux
Jmat = Flux_calc(u, mu, c)

''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of forces (test fxn: u)
# Res_1: Pressure variable (test fxn: p)
# Res_2: Balance of mass   (test fxn: mu)
# Res_3: Auxiliary variable (test fxn: c)

# Time step field, constant  
dk = Constant(dt)

# The weak form for the equilibrium equation
Res_0 = inner(Tmat, grad(u_test) )*dx

# The weak form for the auxiliary pressure variable definition
Res_1 = dot((p*Je/Kbulk + ln(Je)) , p_test)*dx

# The weak form for the mass balance of solvent      
Res_2 = dot((c - c_old)/dk, mu_test)*dx \
        -  Omega*dot(Jmat , grad(mu_test) )*dx

# The weak form for the concentration
fac = 1/(1+c)
fac1 =  mu - ( ln(1.0-fac)+ fac + chi*fac*fac)
fac2 = - (Omega*Je/RT)*p 
fac3 = fac1 + fac2 
#
Res_3 = dot(fac3, c_test)*dx
        
# Total weak form
Res = Res_0 + Res_1 + Res_2 + Res_3

# Automatic differentiation tangent:
a = derivative(Res, w, dw)

'''''''''''''''''''''''
BOUNDARY CONDITIONS
'''''''''''''''''''''''
# Surface labels form 
# Physical Surface("right_bot", 29)
# //+
# Physical Surface("left_top", 30)
# //+
# Physical Surface("inner_surf", 31)
# //+
# Physical Surface("z_bot", 32)
# //+
# Physical Surface("z_top", 33)
#
encastre = Constant((0.0, 0.0, 0.0))
#
bcs_1 = DirichletBC(ME.sub(0).sub(1), 0, facets, 29)     # fix u_1  
bcs_2 = DirichletBC(ME.sub(0).sub(0), 0, facets, 30)     # fix u_2  on face3
bcs_3 = DirichletBC(ME.sub(0), encastre, facets, 31)      # fix all u on face2
bcs_4 = DirichletBC(ME.sub(2), muAmp, facets, 32)        # chem. pot. on face5
bcs_5 = DirichletBC(ME.sub(2), muAmp, facets, 33)        # chem. pot. on face6
# 
bcs = [bcs_1, bcs_2, bcs_3, bcs_4, bcs_5]

"""
SETUP NONLINEAR PROBLEM
"""
GelProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(GelProblem)

#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-6
prm['newton_solver']['relative_tolerance'] = 1.e-6
prm['newton_solver']['maximum_iterations'] = 30

'''''''''''''''''''''
 SET UP OUTPUT FILES
'''''''''''''''''''''
# Output file setup
file_results = XDMFFile("results/gel_thin_corona.xdmf")
# "Flush_output" permits reading the output during simulation
# (Although this causes a minor performance hit)
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Function space for projection of results
W2 = FunctionSpace(mesh, U2) # Vector space for visualization  
W = FunctionSpace(mesh, P1)   # Scalar space for visualization 
     
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
     
# Write initial state to XDMF file
writeResults(t=0.0) 

print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time 
startTime = datetime.now()

# Give the step a descriptive name
step = "Swell"

# Set increment counter to zero
ii = 0

while (t < Ttot):
    
    # smaller time-step needed for initiation of buckling
    if (t>450 and t<1000): 
        dt = 5
        dk.assign(dt)
    else:
        dt = 25
        dk.assign(dt)
        
    # increment time
    t += dt
    
    # Increment counter
    ii += 1

    # Update time variables in time-dependent BCs
    muAmp.t = t

    # Solve the problem
    (iter, converged) = solver.solve()  
    
    # Write output to *.xdmf file
    writeResults(t)
    
    # Update DOFs for next step
    w_old.vector()[:] = w.vector()
    
    
    # Print progress of calculation periodically
    if t%1 == 0:      
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
        print("Simulation Time: {} s".format(round(t,2)))
        print()     
        
# End analysis
print("-----------------------------------------")
print("End computation")                 
# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("------------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("------------------------------------------")
 
