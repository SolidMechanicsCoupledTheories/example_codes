"""
Code for Gels

Plane strain growth  of a  circular tumor

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

Lallit Anand   
anand@mit.edu   

September 2023
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

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''
# Matrix edge length
L0  = 50 
# Ellipse major axis
a0 = 2 
# Ellipse minor axis
b0 = 2 


# Initialize an empty mesh object
mesh = Mesh()

# Read the .xdmf  file data into mesh object
with XDMFFile("meshes/spherical_inclusion.xdmf") as infile:
    infile.read(mesh)
    
# Extract initial mesh coords
x = SpatialCoordinate(mesh)
    
# Read mesh boundary facets.
# Specify parameters of 1-D mesh function object
mvc_1d = MeshValueCollection("size_t", mesh, 1)
# Read the   *.xdmf  data into mesh function object
with XDMFFile("meshes/facet_spherical_inclusion.xdmf") as infile:
    infile.read(mvc_1d, "name_to_read")
    
# Store boundary facets data
facets = cpp.mesh.MeshFunctionSizet(mesh, mvc_1d)

# Gmsh labels
# /+
# Physical Curve("left", 13) 
# //+
# Physical Curve("right", 14) 
# //+
# Physical Curve("bottom", 15)  
# //+
# Physical Curve("top", 16) 
# //+
# Physical Curve("inclusion_curve", 17) 

#Identify the  two subdomains for the different materials
#
tol = 1E-3
# inclusion
class Omega_0(SubDomain):
    def inside(self, x, on_boundary):
        return  (x[0]**2/a0**2+ x[1]**2/b0**2) <= 1 + tol
                               
# Matrix
class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        return   (x[0]**2/a0**2+ x[1]**2/b0**2) >= 1 - tol

materials = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
subdomain_0 = Omega_0()
subdomain_1 = Omega_1()
subdomain_0.mark(materials, 0)
subdomain_1.mark(materials, 1)


"""
User defined expression for defining different materials
"""
class mat(UserExpression): 
    def __init__(self, materials, mat_0, mat_1, **kwargs):
        super().__init__(**kwargs)
        self.materials = materials
        self.k_0 = mat_0
        self.k_1 = mat_1
        
    def eval_cell(self, values, x, cell):
        if self.materials[cell.index] == 0:
            values[0] = self.k_0
        else:
            values[0] = self.k_1
            
    def value_shape(self):
        return ()     
    
# Volume measure over diffrent subdomains, if needed    
dx = Measure('dx', domain=mesh, subdomain_data=materials)


'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
# Set the locking stretch to a large number to model a Neo-Hookean material

# Swellable inclusion
#
G_0      = Constant(0.5)        # Shear modulus, kPa
K_0      = Constant(1000*G_0)     # Bulk modulus, KPa
chi_0    = Constant(0.1)        # Flory-Huggins mixing parameter
Omega_0  = Constant(1.00e5)     # Molar volume of fluid
D_0      = Constant(5.0E-3)    # Diffusivity

#
# Non_swellable matrix
#
G_1      = Constant(0.3)        # Shear modulus, kPa
K_1      = Constant(1000*G_1)     # Bulk modulus, KPa
chi_1    = Constant(0.9)        # Flory-Huggins mixing parameter
Omega_1  = Constant(0.1)        # Molar volume 
D_1      = Constant(5.0E-3)    # Diffusivity
#------------------------------------------------------------------------------
#            
Gshear_0 = mat(materials, G_0, G_1, degree=1) 
lambdaL  = Constant(1000)            # Locking stretch large number
Kbulk    = mat(materials, K_0, K_1, degree=1) 
# 
Omega    = mat(materials, Omega_0, Omega_1, degree=1)
D        = mat(materials, D_0, D_1, degree=1)
chi      = mat(materials, chi_0, chi_1, degree=1)
#-----------------------------------------------------------------------------
theta0   = Constant(298)            # Reference temperature
R_gas    = Constant(8.3145e6)       # Gas constant
#-----------------------------------------
phi0     = Constant(0.999)          # Initial polymer volume fraction
RT       = 8.3145e6*theta0 

"""
Simulation time-control related params
"""
t    = 0.0      # initialization of time
Ttot = 3600     # total simulation time 
ttd  = 300      # Decay time constant
dt   = 50       # Fixed step size

# Boundary condition expression for increasing  the chemical potential
# 
mu0 = ln(1.0-phi0) + phi0 # Initialize chemical potential, corresponding to nearly dry polymer.
muAmp = Expression(("mu0*exp(-t/td)"),
                mu0 = mu0, td = ttd, t = 0.0, degree=1)

'''''''''''''''''''''
Function spaces
'''''''''''''''''''''
# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For  pressure,  normalized chemical potential,
                                                   # and  normalized species concentration
#
TH = MixedElement([U2, P1, P1, P1]) # Taylor-Hood style mixed element
ME = FunctionSpace(mesh, TH)        # Total space for all DOFs

# Define trial functions
w = Function(ME)
u, p, mu, c = split(w)  # displacement u, p, pressure, chemical potential  mu, concentration c

# A copy of functions to store values in the previous step
w_old = Function(ME)
u_old,   p_old,  mu_old,  c_old  = split(w_old)   

# Define test functions
w_test = TestFunction(ME)                
u_test, p_test, mu_test, c_test  = split(w_test)   

# Define trial functions needed for automatic differentiation
dw = TrialFunction(ME)             

# Assign initial value of normalized chemical potential mu0
init_mu = Constant(mu0) 
mu_init = interpolate(init_mu,ME.sub(2).collapse())
assign(w_old.sub(2),mu_init)
assign(w.sub(2), mu_init)

# Assign initial  value of  normalized concentration c0
c0 = 1/phi0 - 1
#
init_c = Constant(c0)
c_init = interpolate(init_c, ME.sub(3).collapse())
assign(w_old.sub(3), c_init)
assign(w.sub(3), c_init)

'''''''''''''''''''''
SUBROUTINES
'''''''''''''''''''''
'''
For plane strain:
'''
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
    Id = Identity(dim)          # Identity tensor
    F  = Id + grad(u)            # 2D Deformation gradient
    return as_tensor([[F[0,0], F[0,1], 0],
                  [F[1,0], F[1,1], 0],
                  [0, 0, 1]]) # Full pe F


def lambdaBar_calc(u):
    F = F_pe_calc(u)
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
    F = F_pe_calc(u)  
    detF = det(F)   
    #
    detFs = 1.0 + c
    Je    = (detF/detFs)
    return   Je    

#-----------------------------------------------------------------------------
# Piola stress

def Piola_calc(u,p):
    F     = F_pe_calc(u)
    zeta  = zeta_calc(u)
    zeta0 = zeta0_calc()
    Tmat = (zeta*F - zeta0*inv(F.T) ) - J*p*inv(F.T)/Gshear_0
    return Tmat

#------------------------------------------------------------------------------  
# Normalized species flux
def Flux_calc(u, mu, c):
    F = F_pe_calc(u) 
    #
    Cinv = inv(F.T*F) 
    #
    Mob = (D*c)/(Omega*RT)*Cinv
    #
    Jmat = - RT* Mob * pe_grad_scalar(mu)
    return Jmat

'''''''''''''''''''''''''''''
Kinematics and constitutive relations
'''''''''''''''''''''''''''''
# Kinematics
F = F_pe_calc(u)
J = det(F)  # Total volumetric jacobian

# Elastic volumetric Jacobian
Je     = Je_calc(u,c)                    
Je_old = Je_calc(u_old,c_old)

#  Piola stress
Tmat = Piola_calc(u, p)

# Calculate the Species flux
Jmat = Flux_calc(u, mu, c)


''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of forces (test fxn: u)
# Res_1: Pressure variable (test fxn: p)
# Res_2: Balance of mass   (test fxn: mu)
# Res_3: Auxiliary variable (test fxn: c)

# Time step field, constant within body
dk = Constant(dt)

# The weak form for the equilibrium equation
Res_0 = inner(Tmat , pe_grad_vector(u_test) )*dx

# The weak form for the auxiliary pressure variable 
Res_1 = dot((p*Je/Kbulk + ln(Je)) , p_test)*dx

# The weak form for the mass balance of solvent      
Res_2 = dot((c - c_old)/dk, mu_test)*dx \
        -  Omega*dot(Jmat , pe_grad_scalar(mu_test) )*dx      

# The weak form for the concentration
fac = 1/(1+c)
fac1 =  mu - ( ln(1.0-fac)+ fac + chi*fac*fac)
fac2 = - (Omega*Je/RT)*p 
fac3 = fac1 + fac2 
Res_3 = dot(fac3, c_test)*dx
        
# Total weak form
Res = Res_0 + Res_1 + Res_2 + Res_3

# Automatic differentiation tangent:
a = derivative(Res, w, dw)
   


'''''''''''''''''''''
 SET UP OUTPUT FILES
'''''''''''''''''''''
# Output file setup
file_results = XDMFFile("results/gel_circular_tumor_growth.xdmf")
# "Flush_output" permits reading the output during simulation
# (Although this causes a minor performance hit)
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True


# Function space for projection of results
W2 = FunctionSpace(mesh, U2)   # Vector space for visualization  
W = FunctionSpace(mesh,  P1)   # Scalar space for visualization 
     
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
      
      # # Visualize some Piola stress components
      # P11_Vis = project(Tmat[0,0]*1000,W)
      # P11_Vis.rename("P11, Pa","")
      # P22_Vis = project(Tmat[1,1]*1000,W)
      # P22_Vis.rename("P22, Pa","")    
      # P33_Vis = project(Tmat[2,2]*1000,W)
      # P33_Vis.rename("P33, Pa","")        
      
      # Visualize the Mises stress  
      T    = Tmat*F.T/J
      T0   = T - (1/3)*tr(T)*Identity(3)
      #
      Mises = sqrt((3/2)*inner(T0, T0))
      Mises_Vis = project(Mises*1000,W)
      Mises_Vis.rename("Mises, Pa"," ")   
      
      # Visualize some CUachy stress components
      T11_Vis = project(T[0,0]*1000,W)
      T11_Vis.rename("T11, Pa","")
      T22_Vis = project(T[1,1]*1000,W)
      T22_Vis.rename("T22, Pa","")    
      T33_Vis = project(T[2,2]*1000,W)
      T33_Vis.rename("T33, Pa","")        
      
      # Visualize the Shear modulus 
      Gshear_Vis = project(Gshear_0, W)
      Gshear_Vis.rename("Gshear",  "")
      
      # Visualize the chi parameter
      chi_Vis = project(chi, W)
      chi_Vis.rename("chi",  "")
      
      # Visualize the chi parameter
      D_Vis = project(D, W)
      D_Vis.rename("Diffusivity",  "")
      
      # Visualize the Omega parameter
      Omega_Vis = project(Omega, W)
      Omega_Vis.rename("Omega",  "")
      
      # Visualize the Kbulk
      Kbulk_Vis = project(Kbulk, W)
      Kbulk_Vis.rename("Kbulk",  "")
     
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
      # file_results.write(P11_Vis, t)  
      # file_results.write(P22_Vis, t)    
      # file_results.write(P33_Vis, t)  
      file_results.write(T11_Vis, t)  
      file_results.write(T22_Vis, t)    
      file_results.write(T33_Vis, t)  
      file_results.write(Mises_Vis, t)
      file_results.write(Gshear_Vis, t) 
      file_results.write(chi_Vis, t)     
      file_results.write(D_Vis, t)   
      file_results.write(Omega_Vis, t)   
      file_results.write(Kbulk_Vis, t)           
     
# Write initial state to XDMF file
writeResults(t=0.0)  



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
# Gmsh labels
# /+
# Physical Curve("left", 13) 
# //+
# Physical Curve("right", 14) 
# //+
# Physical Curve("bottom", 15)  
# //+
# Physical Curve("top", 16) 
# //+
# Physical Curve("inclusion_curve", 17) 

bcs_1 = DirichletBC(ME.sub(0).sub(0), Constant(0), facets, 13)
bcs_2 = DirichletBC(ME.sub(0).sub(1), Constant(0), facets, 13)
bcs_3 = DirichletBC(ME.sub(0).sub(0), Constant(0), facets, 14)
bcs_4 = DirichletBC(ME.sub(0).sub(1), Constant(0), facets, 14)
bcs_5 = DirichletBC(ME.sub(0).sub(1), Constant(0), facets, 15)
bcs_6 = DirichletBC(ME.sub(0).sub(1), Constant(0), facets, 15)
#
bcs_7 = DirichletBC(ME.sub(2), muAmp, facets, 17) 
#
bcs = [bcs_1, bcs_2, bcs_3, bcs_4, bcs_5, bcs_6, bcs_7]


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



# Initialize an  array for storing results
siz       = 100000 
Forc      = np.zeros([siz+1,3])
Forc[0,1] = a0    # initial  value of major axis  of ellipse
Forc[0,2] = b0    # initial  value of minor axis  of ellipse


# Set increment counter to zero
ii = 0

while (t < Ttot):
    
    # # smaller time-step needed for initiation of buckling
    # if (t>2000 and t<3600): 
    #     dt = 10
    #     dk.assign(dt)
    # else:
    #     dt = 50
    #     dk.assign(dt)
    
    # increment time
    t += dt
    
    # Increment counter
    ii += 1

    # Update time variables in the time-dependent BCs
    muAmp.t = t

    # Solve the problem
    (iter, converged) = solver.solve()  
    
    # Write output to *.xdmf file
    writeResults(t)
    
    # Update DOFs for next step
    w_old.vector()[:] = w.vector()
    
    # Write time histories 
    #
    #
    Forc[ii,0] = t # store the time 
    #
    Forc[ii,1] = a0 + w.sub(0).sub(0)(a0, 0) - w.sub(0).sub(0)(0,0) #  major axis
    #
    Forc[ii,2] = b0 + w.sub(0).sub(1)(0, b0) - w.sub(0).sub(1)(0,0) # minor axis
    #
     
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

'''''''''''''''''''''
    VISUALIZATION
'''''''''''''''''''''

# Set up font size, initialize colors array
font = {'size'   : 14}
plt.rc('font', **font)
#
prop_cycle = plt.rcParams['axes.prop_cycle']
colors     = prop_cycle.by_key()['color']

# Only plot as far as time history data
ind = np.argmax(Forc[:,1])
   
# Create figure for stress-stetch curve.
#
fig = plt.figure() 
ax=fig.gca()  
plt.plot( Forc[0:ind,0], Forc[0:ind,1], 'b', linewidth=1.0, marker='.', label= "major semi-axis")
plt.plot( Forc[0:ind,0], Forc[0:ind,2], 'r', linewidth=1.0, marker='.', label= "minor semi-axis")
#-----------------------------------------------------
plt.grid(linestyle="--", linewidth=0.5, color='b')
ax.set_xlabel("t, s",size=14)
ax.set_ylabel("Major  and minor axes, mm",size=14)
#ax.set_title("Major axis a versus time curve", size=14, weight='normal')
plt.legend()
#
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

plt.show()
# Save figure to file
fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/circular_tumor_growth.png", dpi=600) 

