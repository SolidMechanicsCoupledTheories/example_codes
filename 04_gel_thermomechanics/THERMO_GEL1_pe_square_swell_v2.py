"""
Plane strain code for thermoresponsive hydrogels

Degrees of freedom:
    > vectorial mechanical displacement: u
    > scalar pressure: p
    > scalar chemical potential of solvent: we use normalized  mu = mu/RT
    > species concentration:  we use normalized  c= Omega*cmat
    > temperature: theta
    
Units:
#
Length: mm
Mass: tonne (1000 kg)
Time: s
Mass density: tonne/mm^3
Force: N
Stress: MPa 
Energy: mJ
Temperature: K
Amount of substance: mol
Species concentration: mol/mm^3
Chemical potential: mJ/mol
Molar volume: mm^3/mol
Species diffusivity: mm^2/s
Thermal expansion coefficient: #/K
Specific heat: mJ/(mm^3 K)
Thermal conductivity: mW/(mm K)
Boltzmann Constant: 1.38E-20 mJ/K
Gas constant: 8.314E3  mJ/(mol K)

Lallit Anand   and Eric Stewart
anand@mit.edu     ericstew@mit.edu

September 2023
"""

# Fenics-related packages
from dolfin import *
# Numerical array package
import numpy as np
# 
from ufl import tanh
#
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
R0 = 2.5  # domain radius
H0 = 2.5  # domian height
# Last two numbers below are the number of elements in the two directions
mesh = RectangleMesh(Point(0, 0), Point(R0, H0), 15, 15, "crossed")

x = SpatialCoordinate(mesh)

# Identify the boundary entities of mesh
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],R0) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],H0) and on_boundary 
    
# Identify the bottom left corner       
def Ground(x, on_boundary):
        return near(x[0],0) and near(x[1], 0)

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
n = FacetNormal(mesh)

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
k_B      = Constant(1.38E-20)              # Boltzmann's constant
R_gas    = Constant(8.3145E3)              # Gas constant
theta0   = Constant(298)                   # Initial temperature
Gshear_0 = Constant(0.3)                   # Ground sate shear modulus
N_R      = Constant(Gshear_0/(k_B*theta0)) # Number polymer chains per unit ref. volume
lambdaL  = Constant(5.2)                   # Locking stretch
Kbulk    = Constant(1000.0)       # Bulk modulus
#
Omega    = Constant(1.0E5)                 # Molar volume of fluid
D        = Constant(5.0E-3)                # Diffusivity
chi_L    = Constant(0.1)                   # Flory-Huggins mixing parameter low value
chi_H    = Constant(0.7)                   # Flory-Huggins mixing parameter high value
theta_T  = Constant(307)                   # Transition temperature
Delta_T  = Constant(5.0)                   # Transition temperature width
alpha    = Constant(70.0E-6)              # Coefficient of thermal expansion
c_v      = Constant(4.18)                  # Specific heat
k_therm  = Constant(0.53)                # Thermal conductivity
#
phi0    = Constant(0.999)                  # Initial polymer volume fraction

"""
Simulation time-control related parameters
"""
t          = 0.0        # initialization of time
ttd        = 300        # time constant for increasing mu
step1_time = 3600*2    
# 
tRamp      = 3600*2     # ramp time for increasing/decreasing temperature
#
step2_time = 3600*3    # ramp time + hold time for heating step  
#
step3_time = 3600*3   # ramp time + hold time for cooling step
#
Ttot = step1_time + step2_time + step3_time 
dt   = 50          

# Boundary condition expressions for increasing  the chemical potential and temperature
# 
mu0 = ln(1.0-phi0) + phi0 # Initialize chemical potential, corresponding to nearly dry polymer.
muRamp = Expression("mu0*exp(-t/td)",
                mu0 = mu0, td = ttd, t = 0.0, degree=1)

'''''''''''''''''''''
Function spaces
'''''''''''''''''''''
# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For pressure,
                                                   # normalized chemical potential,
                                                   # normalized species concentration, and
                                                   # temperature                                                  
#
TH = MixedElement([U2, P1, P1, P1, P1]) # Taylor-Hood style mixed element
ME = FunctionSpace(mesh, TH)            # Total space for all DOFs

# Define actual functions with the required DOFs
w = Function(ME)
u, p, mu, c, theta = split(w)  # displacement u, pressure p, chemical potential  mu,  concentration c, temperature theta

# A copy of functions to store values in the previous step for time-stepping
w_old = Function(ME)
u_old, p_old,  mu_old,  c_old, theta_old = split(w_old)   

# Define test functions 
w_test = TestFunction(ME)                
u_test, p_test,  mu_test, c_test, theta_test = split(w_test)   

#Define trial functions needed for automatic differentiation
dw = TrialFunction(ME)                  

# Initialize chemical potential, corresponding to nearly dry polymer
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

# Assign initial  temperature  theta0
init_theta = Constant(theta0)
theta_init = interpolate(init_theta, ME.sub(4).collapse())
assign(w_old.sub(4), theta_init)
assign(w.sub(4), theta_init)
 
'''''''''''''''''''''
SUBROUTINES
'''''''''''''''''''''
'''
For plane strain:
'''
# Special gradient operators for plane strain test functions 
#
# Gradient of vector field u   
def pe_grad_vector(u):
    grad_u = grad(u)
    return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, 0]]) 

# Gradient of scalar field y
# (just need an extra zero for dimensions to work out)
def pe_grad_scalar(y):
    grad_y = grad(y)
    return as_vector([grad_y[0], grad_y[1], 0.])

# Plane strain deformation gradient 
def F_pe_calc(u):
    dim = len(u)
    Id = Identity(dim)           # 2D identity tensor
    F  = Id + grad(u)            # 2D Deformation gradient
    return as_tensor([[F[0,0], F[0,1], 0],
                  [F[1,0], F[1,1], 0],
                  [0, 0, 1]]) # Full pe F

#  Elastic deformation gradient Fe
def Fe_calc(u,c):
    F = F_pe_calc(u)      
    J = det(F)             
    #
    Js = 1.0 + c          
    Fs = Js**(1/3)*Identity(3) 
    #
    Fe = F*inv(Fs)
    return   Fe    

# lambdaBar
def lambdaBar_calc(u):
    F = F_pe_calc(u)
    C = F.T*F
    I1 = tr(C)
    lambdaBar = sqrt(I1/3.0)
    return lambdaBar

# zeta
def zeta_calc(u):
    lambdaBar = lambdaBar_calc(u)
    # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
    # This is sixth-order accurate.
    z    = lambdaBar/lambdaL
    z    = conditional(gt(z,0.95), 0.95, z) # Prevent the function from blowing up
    beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
    zeta = (lambdaL/(3*lambdaBar))*beta
    return zeta

# zeta0
def zeta0_calc():
    # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
    # This is sixth-order accurate.
    z    = 1/lambdaL
    beta0 = z*(3.0 - z**2.0)/(1.0 - z**2.0)
    zeta0 = (lambdaL/3)*beta0
    return zeta0

# chi-parameter
def chi_calc(theta):
     chi = 0.5*(chi_L + chi_H)- 0.5*(chi_L - chi_H)* tanh((theta-theta_T)/Delta_T)
     return chi

# Stress-temperature modulus
def MH1_calc(u,c):
    Id = Identity(3)
    Fe = Fe_calc(u,c)
    Je = det(Fe)
    Ce = Fe.T*Fe
    Js = 1 + c
    zeta = zeta_calc(u)
    zeta0 = zeta0_calc()
    fac1 = N_R * k_B * ( zeta * Js**(2/3) * Id - zeta0 * inv(Ce) )
    fac2 = (3*Kbulk*alpha/Je) * inv(Ce)
    MH1 = 0.5*(fac1 +fac2)
    return MH1

# Chemical potential-temperature modulus
def MH2_calc(u,c):
    Id = Identity(3)
    F = F_pe_calc(u)
    C = F.T*F
    Js = 1 + c
    zeta  = zeta_calc(u)
    zeta0 = zeta0_calc()
    chi   = chi_calc(theta)
    #
    fac1 = R_gas*( ln(c/Js) + 1/Js + chi/Js**2 )
    fac2 = (Omega*N_R*k_B/Js)*( zeta*tr(C)/3 - zeta0 )
    MH2 = fac1+ fac2
    return MH2


# Normalized Piola stress 
def Piola_calc(u, p, theta):
    F = F_pe_calc(u)
    #
    J = det(F)
    #
    zeta = zeta_calc(u)
    #
    zeta0 = zeta0_calc()
    #
    Gshear0  = N_R * k_B * theta
    #
    Tmat = (zeta*F - zeta0*inv(F.T) ) - (J * p/Gshear0) * inv(F.T) 
    return Tmat


# Species flux
def Flux_calc(u, mu, c, theta):
    F = F_pe_calc(u) 
    #
    Cinv = inv(F.T*F) 
    #
    RT = R_gas * theta
    #
    Mob = (D*c)/(Omega*RT)*Cinv # Mobility tensor
    #
    Jmat = - RT* Mob * pe_grad_scalar(mu)
    return Jmat



#  Heat flux
def Heat_flux_calc(u, theta):
    F  = F_pe_calc(u) 
    J = det(F)         
    #
    Cinv = inv(F.T*F) 
    #
    Tcond = J * k_therm * Cinv # Thermal conductivity tensor
    #
    Qmat = - Tcond * pe_grad_scalar(theta)
    return Qmat



'''''''''''''''''''''''''''''
Kinematics and constitutive relations
'''''''''''''''''''''''''''''
# Kinematics
F = F_pe_calc(u)
J = det(F)  

# Fe 
Fe     = Fe_calc(u,c)                    
Fe_old = Fe_calc(u_old,c_old)

# Je
Je = det(Fe)

# Ce
Ce = Fe.T * Fe
Ce_old = Fe_old.T * Fe_old

#  Piola stress
Tmat = Piola_calc(u, p, theta)

# Species flux
Jmat = Flux_calc(u, mu, c, theta)

# Heat flux
Qmat = Heat_flux_calc(u, theta)

# Heat-coupling terms
MH1 = MH1_calc(u,c)
MH2 = MH2_calc(u,c)

# RT
RT      = R_gas*theta

# chi-parameter
chi = chi_calc(theta)

''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of forces (test fxn: u)
# Res_1: Pressure (test fxn: p)
# Res_2: Balance of mass   (test fxn: mu)
# Res_3: Concentration     (test fxn:c)
# Res_4: Tempearture       (test fxn:theta)

# Time step field, constant within body
dk = Constant(dt)

# The weak form for the equilibrium equation
Res_0 = inner(Tmat , pe_grad_vector(u_test) )*dx

# The weak form for the  pressure
fac_p =   (ln(Je) - 3*alpha*(theta-theta0))
#
Res_1 = dot((p*Je/Kbulk + fac_p) , p_test)*dx
      
# The weak form for the mass balance of solvent      
Res_2 = dot((c - c_old)/dk, mu_test)*dx \
        -  Omega*dot(Jmat , pe_grad_scalar(mu_test) )*dx      

# The weak form for the concentration
fac = 1/(1+c)
#
fac1 =  mu - ( ln(1.0-fac)+ fac + chi*fac*fac ) -(Omega*Je/RT)*p  
#
Res_3 = dot(fac1, c_test)*dx

# The weak form for the the heat equation
tfac1 = theta * inner(MH1, (Ce-Ce_old))
tfac2 = theta * MH2*(c - c_old)/Omega
tfac3 = dk * RT * inner(Jmat, pe_grad_scalar(mu))
tfac4 = tfac1 + tfac2 - tfac3
#
Res_4 = dot( c_v*(theta - theta_old), theta_test)*dx \
        -  dk* dot(Qmat , pe_grad_scalar(theta_test) )*dx   \
        -   dot(tfac4, theta_test)*dx 
        
# Total weak form
Res = Res_0 + Res_1 + Res_2 + Res_3 + Res_4

# Automatic differentiation tangent:
a = derivative(Res, w, dw)

'''''''''''''''''''''
 SET UP OUTPUT FILES
'''''''''''''''''''''
# Output file setup
file_results = XDMFFile("results/thermogel_pe_swell.xdmf")
# "Flush_output" permits reading the output during simulation
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Function space for projection of results
W2 = FunctionSpace(mesh, U2)   # Vector space for visualization 
W  = FunctionSpace(mesh,P1)    # Scalar space for visualization 

# Subroutine for writing output to file
def writeResults(t):
    
    # Variable casting and renaming
    # u, p, mu, c, theta = split(w)
    
    # Visualize displacement
    u_Vis = project(u, W2)
    u_Vis.rename("disp"," ")
    
    # Visualize the pressure
    p_Vis = project(p, W)
    p_Vis.rename("p"," ")
    
    # Visualize  normalized chemical potential
    mu_Vis = project(mu,W)
    mu_Vis.rename("mu"," ")   
    
    # Visualize  normalized concentration
    c_Vis = project(c,W)
    c_Vis.rename("c"," ")
    
    # Visualize  temperature
    theta_Vis = project(theta, W)
    theta_Vis.rename("theta"," ")

    # Visualize polymer volume fraction phi
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
      
    # Visualize the Mises stress 
    T    = Tmat*F.T/J
    T0   = T - (1/3)*tr(T)*Identity(3)
    #
    Mises = sqrt((3/2)*inner(T0, T0))
    Mises_Vis = project(Mises*1000,W)
    Mises_Vis.rename("Mises, kPa"," ")   
    
    # Visualize some Cauchy stress components
    T11_Vis = project(T[0,0]*1000,W)
    T11_Vis.rename("T11, kPa","")
    T22_Vis = project(T[1,1]*1000,W)
    T22_Vis.rename("T22, kPa","")    
    T33_Vis = project(T[2,2]*1000,W)
    T33_Vis.rename("T33, kPa","")         
      
    # Write field quantities of interest to file
    file_results.write(u_Vis, t)
    file_results.write(p_Vis, t)
    file_results.write(mu_Vis, t)
    file_results.write(c_Vis, t)
    file_results.write(theta_Vis, t)
    file_results.write(phi_Vis, t)
    file_results.write(J_Vis, t)  
    file_results.write(lambdaBar_Vis, t)
    file_results.write(Je_Vis, t)
    file_results.write(Mises_Vis, t)
    file_results.write(T11_Vis, t)  
    file_results.write(T22_Vis, t)    
    file_results.write(T33_Vis, t)  

# Write initial values
writeResults(t=0.0)


"""
Start simulation
"""
print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time 
startTime = datetime.now()

# Give the step a descriptive name
step = "Swell"

'''''''''''''''''''''''
Dirichlet  Boundary Conditions
'''''''''''''''''''''''
# Left().mark(facets,   1)
# Bottom().mark(facets, 2)
# Top().mark(facets,    3)
# Right().mark(facets,  4)

# Boundary condition definitions
bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 1)  # u1 fix - Left  
bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 2)  # u2 fix - Bottom
#
bcs_3 = DirichletBC(ME.sub(2), muRamp, facets, 3)    # chem. pot. - Right
bcs_4 = DirichletBC(ME.sub(2), muRamp, facets, 4)    # chem. pot. - Top
#
bcs_5 = DirichletBC(ME.sub(4), theta0, facets, 3) # Temperature - Right
bcs_6 = DirichletBC(ME.sub(4), theta0, facets, 4) # Temperature - Top

#
bcs = [bcs_1, bcs_2, bcs_3, bcs_4, bcs_5, bcs_6]

"""
SETUP NONLINEAR PROBLEM
"""
GelProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(GelProblem)
#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-8
prm['newton_solver']['maximum_iterations'] = 30

# Set increment counter to zero
ii = 0

while (t < step1_time):
    
    # increment time
    t += dt
    
    # Increment counter
    ii += 1

    # update time variables in time-dependent BCs
    muRamp.t    = t

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
        print("Simulation Time: {} h".format(round(t/3600,2)))
        print()     
   
   
print("------------------------------------")
print("Step 2")
print("------------------------------------")

# Give the step a descriptive name
step = "Heat & De-swell"

thetaRamp = Expression("min(theta0 + deltaTheta*(t-step1_time)/tRamp, theta0 + deltaTheta)", \
             theta0 = 298, deltaTheta = 25, t= 0.0, step1_time=step1_time, tRamp = tRamp, degree=1)

'''''''''''''''''''''''
Dirichlet  Boundary Conditions
'''''''''''''''''''''''
# Left().mark(facets,   1)
# Bottom().mark(facets, 2)
# Top().mark(facets,    3)
# Right().mark(facets,  4)

# Boundary condition definitions
bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 1)  # u1 fix - Left  
bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 2)  # u2 fix - Bottom
#
bcs_3 = DirichletBC(ME.sub(2), muRamp, facets, 3)    # chem. pot. - Right
bcs_4 = DirichletBC(ME.sub(2), muRamp, facets, 4)    # chem. pot. - Top
#
bcs_5 = DirichletBC(ME.sub(4), thetaRamp, facets, 3) # Temperature - Right
bcs_6 = DirichletBC(ME.sub(4), thetaRamp, facets, 4) # Temperature - Top

#
bcs = [bcs_1, bcs_2, bcs_3, bcs_4, bcs_5, bcs_6]

"""
SETUP NONLINEAR PROBLEM
"""
GelProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(GelProblem)
#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-8
prm['newton_solver']['maximum_iterations'] = 30

# Set increment counter to zero
ii = 0

while (t < step1_time + step2_time):
    
    # increment time
    t += dt
    
    # Increment counter
    ii += 1

    # update time variables in time-dependent BCs
    muRamp.t    = t
    thetaRamp.t = t 

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
        print("Simulation Time: {} h".format(round(t/3600,2)))
        print()     
   
print("------------------------------------")
print("Step 3")
print("------------------------------------")

# Give the step a descriptive name
step = "Cool & Re-swell"

thetaRamp2 = Expression("max(theta0 + deltaTheta*(1 - (t - (step1_time + step2_time))/tRamp),theta0)", \
             theta0 = float(theta0), deltaTheta = 25, t= 0.0, step1_time=step1_time,\
                 step2_time = step2_time, tRamp = tRamp, degree=1)
    
'''''''''''''''''''''''
Dirichlet  Boundary Conditions
'''''''''''''''''''''''
# Left().mark(facets,   1)
# Bottom().mark(facets, 2)
# Top().mark(facets,    3)
# Right().mark(facets,  4)

# Boundary condition definitions
bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 1)  # u1 fix - Left  
bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 2)  # u2 fix - Bottom
#
bcs_3 = DirichletBC(ME.sub(2), muRamp, facets, 3)    # chem. pot. - Right
bcs_4 = DirichletBC(ME.sub(2), muRamp, facets, 4)    # chem. pot. - Top
#
bcs_5 = DirichletBC(ME.sub(4), thetaRamp2, facets, 3) # Temperature - Right
bcs_6 = DirichletBC(ME.sub(4), thetaRamp2, facets, 4) # Temperature - Top

#
bcs = [bcs_1, bcs_2, bcs_3, bcs_4, bcs_5, bcs_6]

"""
SETUP NONLINEAR PROBLEM
"""
GelProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(GelProblem)
#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-8
prm['newton_solver']['maximum_iterations'] = 30

# Set increment counter to zero
ii = 0

while (t < Ttot):
    
    # increment time
    t += dt
    
    # Increment counter
    ii += 1

    # update time variables in time-dependent BCs
    muRamp.t    = t
    thetaRamp2.t = t 

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
        print("Simulation Time: {} h".format(round(t/3600,2)))
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
