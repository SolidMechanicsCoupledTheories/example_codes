"""
Code for Cahn-Hilliard phase separation,
with mechanical coupling.

2D axisymmetric ellipsoid lithiation study.


Degrees of freedom:
    > vectorial mechanical displacement: u
    > scalar chemical potential: we use normalized  mu = mu/RT
    > species concentration:  we use normalized  c= Omega*cmat 
    
Units:
#
Length: um
Mass: kg
Time: s
Amount of substance: pmol
Temperature: K
#
Mass density: kg/um^3
Force: uN
Stress: MPa 
Energy: pJ
#
Species concentration: pmol/um^3
Chemical potential: pJ/pmol
Molar volume: um^3/pmol
Species diffusivity: um^2/s
#
Boltzmann Constant: 1.38E-11 pJ/K
Gas constant: 8.314  pJ/(pmol K)

  Eric Stewart      and      Lallit Anand
ericstew@mit.edu            anand@mit.edu

October 2023

"""

# Fenics-related packages
from dolfin import *
# Numerical array package
import numpy as np
# random package for randomized ICs
import random
# Plotting packages
import matplotlib.pyplot as plt
plt.close('all')
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

# spherical particle parameters
re = 0.5    # um, external radius

# Initialize an empty mesh object
mesh = Mesh()

# Read the .xdmf  file data into mesh object
with XDMFFile("meshes/axi_ellipse.xdmf") as infile:
    infile.read(mesh)
    
# Read the 2D subdomain data stored in the *.xdmf file
mvc1d = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("meshes/facet_axi_ellipse.xdmf") as infile:
    infile.read(mvc1d, "name_to_read")
    
# Mesh facets
facets = cpp.mesh.MeshFunctionSizet(mesh, mvc1d)

# Curve labels from Gmsh
#
# //+
# Physical Curve("left", 5)  
# //+
# Physical Curve("bottom", 6)  
# //+
# Physical Curve("outer", 7)  

# Extract initial mesh coords
x = SpatialCoordinate(mesh)
# Define the boundary integration measure "ds".
ds = Measure('ds', domain=mesh, subdomain_data=facets)
# Facet normal
n = FacetNormal(mesh)

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
# Material parameters after Di Leo et al. (2014)
#
Gshear   = Constant(49.8e3*0.1)    # Shear modulus, MPa
Kbulk    = Constant(83e3*0.1)      # Bulk modulus, MPa
# 
Omega    = Constant(4.05)      # Molar volume, um^3/pmol 
D        = Constant(1e-2)      # Diffusivity, um^2/s 
chi      = Constant(3)         # Phase parameter, (-)
cMax     = Constant(2.29e-2)   # Saturation concentration, pmol/um^3
lam      = Constant(5.5749e-1) # Interface parameter, (pJ/pmol) um^2
#
theta0   = Constant(298)     # Reference temperature, K
R_gas    = Constant(8.3145)  # Gas constant, pJ/(pmol K)
RT       = R_gas*theta0 

"""
Simulation time-control related params
"""
t    = 0.0   # initialization of time
Ttot = 30    # total simulation time 
dt   = 0.01  # Initial time step size, here we will use adaptive time-stepping

# initial and final cBar values
cBar_i = 0.005 
cBar_f = 0.995

# corresponding (normalized) chemical potential values
mu_i = ln(cBar_i/(1-cBar_i)) + float(chi)*(1-2*cBar_i)
mu_f = ln(cBar_f/(1-cBar_f)) + float(chi)*(1-2*cBar_f)

# expression for mu "step function"
mu_f_exp = Expression("t <= tRamp ? mu_i + (mu_f - mu_i)*t/tRamp : mu_f ",
                t = 0.0, tRamp = 1.0, mu_i = mu_i, mu_f=mu_f, degree=1)

'''''''''''''''''''''
Function spaces
'''''''''''''''''''''
# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For  normalized chemical potential
                                                   # and  normalized species concentration
#
TH = MixedElement([U2, P1, P1]) # Mixed element
ME = FunctionSpace(mesh, TH)   # Total space for all DOFs

# Define trial functions
w = Function(ME)
u, mu, c = split(w)  # chemical potential  mu, concentration c

# A copy of functions to store values in the previous step
w_old = Function(ME)
u_old, mu_old,  c_old  = split(w_old)   

# Define test functions
w_test = TestFunction(ME)                
u_test, mu_test, c_test  = split(w_test)   

# Define trial functions needed for automatic differentiation
dw = TrialFunction(ME)             

# Initialize chemical potential, corresponding to nearly depleted state
init_mu = Constant(mu_i) 
mu_init = interpolate(init_mu,ME.sub(1).collapse())
assign(w_old.sub(1),mu_init)
assign(w.sub(1), mu_init)

# Assign initial  species normalized concentration c0
init_c = Constant(cBar_i*cMax*Omega)
c_init = interpolate(init_c, ME.sub(2).collapse())
assign(w_old.sub(2),c_init)
assign(w.sub(2), c_init)

'''''''''''''''''''''
SUBROUTINES
'''''''''''''''''''''
'''
For axisymmetric:
'''
# Special gradient operators for axisymmetric test functions 
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

#  Elastic deformation gradient Fe
def Fe_calc(u,c):
    F = F_axi_calc(u)      # = F
    J = det(F)            # = J
    #
    Js = 1.0 + c          # = Js
    Fs = Js**(1/3)*Identity(3) 
    #
    Fe = F*inv(Fs)
    return   Fe    

# The elastic second Piola stress
def Te_calc(u, c):
    Id = Identity(3)
    #
    Fe = Fe_calc(u, c)
    Je = det(Fe)
    Ce = Fe.T*Fe
    #
    Cebar = Je**(-2/3)*Ce
    #
    Te = Je**(-2/3)*Gshear*(Id - (1/3)*tr(Cebar)*inv(Cebar))\
        + Kbulk*Je*(Je-1)*inv(Cebar)
    #
    return Te

# The elastic Mandel stress
def Me_calc(u, c):
    Fe = Fe_calc(u, c)
    Je = det(Fe)
    Ce = Fe.T*Fe
    #
    Te = Te_calc(u, c)
    #
    Me = Ce*Te
    #
    return Me
   
# The first Piola stress     
def Piola_calc(u,c):
    #
    F = F_axi_calc(u)
    J = det(F)
    #
    Fe = Fe_calc(u, c)
    Je = det(Fe)
    #
    Te = Te_calc(u,c)
    #
    T  = Je**(-1)*Fe*Te*inv(Fe)
    #
    TR = J*T*inv(F.T)/Gshear 
    return TR

#------------------------------------------------------------------------------  
# Species flux
def Flux_calc(u, mu, c):
    F = F_axi_calc(u) 
    #
    Cinv = inv(F.T*F) 
    #
    cBar = c/(Omega*cMax) # normalized concentration
    #
    Mob = (D*c)/(Omega*RT)*(1-cBar)*Cinv
    #
    Jmat = - RT* Mob * axi_grad_scalar(mu)
    #
    return Jmat

# Calculate the f^c term
def fc_calc(u, c):
    #
    cBar = c/(Omega*cMax) # normalized concentration
    #
    Me = Me_calc(u,c)
    #
    fc   = RT*(ln(cBar/(1-cBar)) + chi*(1-2*cBar) ) - Omega*((1/3)*tr(Me))
    #
    return fc

'''''''''''''''''''''''''''''
Kinematics and constitutive relations
'''''''''''''''''''''''''''''

# Calculate the normalized concentration cBar
cBar = c/(Omega*cMax) # normalized concentration

# Calculate the Piola stress
TR = Piola_calc(u,c)

# Calculate the Species flux
Jmat = Flux_calc(u, mu, c)

# Calculate the f^c term 
fc = fc_calc(u, c)

''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Equation of motion (test fxn: u)
# Res_1: Balance of mass   (test fxn: mu)
# Res_2: chemical potential (test fxn: c)

# Time step field, constant within body
dk = Constant(dt)

# The weak form for the equation of motion
Res_0 = inner(TR, axi_grad_vector(u_test))*x[0]*dx

# The weak form for the mass balance of mobile species    
Res_1 = dot((c - c_old)/dk, mu_test)*x[0]*dx \
        -  Omega*dot(Jmat , axi_grad_scalar(mu_test) )*x[0]*dx \
        + Omega*dot(mu - mu_f_exp, mu_test)*x[0]*ds(7)

# The weak form for the concentration
Res_2 = dot(mu - fc/RT, c_test)*x[0]*dx \
        -  dot( (lam/RT)*axi_grad_scalar(cBar), axi_grad_scalar(c_test))*x[0]*dx
        
# Total weak form
Res = Res_0 + Res_1 + Res_2

# Automatic differentiation tangent:
a = derivative(Res, w, dw)
   

'''''''''''''''''''''
 SET UP OUTPUT FILES
'''''''''''''''''''''
# Output file setup
file_results = XDMFFile("results/2D_axi_ellipse_lithiation_delithiation.xdmf")
# "Flush_output" permits reading the output during simulation
# (Although this causes a minor performance hit)
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Function space for projection of results
W2 = FunctionSpace(mesh, U2)   # Vector space for visualization 
W = FunctionSpace(mesh,  P1)   # Scalar space for visualization 
     
def writeResults(t):
      # Variable projecting and renaming
      
      # Visualize the displacement
      u_Vis = project(u, W2)
      u_Vis.rename("disp"," ")
      
      # Visualize  the normalized chemical potential
      mu_Vis = project(mu, W)
      mu_Vis.rename("mu"," ")
      
      # Visualize  the normalized concentration
      c_Vis = project(c, W)
      c_Vis.rename("c"," ")
      
      # Visualize cBar
      cBar_Vis = project(cBar, W)
      cBar_Vis.rename("cBar"," ")
      
      # Visualize the Cauchy stress components
      F    = F_axi_calc(u)
      J    = det(F)
      T    = TR*F.T/J
      
      # Visualize some Cauchy stress components
      T11_Vis = project(T[0,0]*Gshear,W)
      T11_Vis.rename("T11, MPa","")
      T22_Vis = project(T[1,1]*Gshear,W)
      T22_Vis.rename("T22, MPa","")    
      T33_Vis = project(T[2,2]*Gshear,W)
      T33_Vis.rename("T33, MPa","") 
   
     # Write field quantities of interest
      file_results.write(u_Vis, t)
      file_results.write(mu_Vis, t)
      file_results.write(c_Vis, t)
      file_results.write(cBar_Vis, t)
      file_results.write(T11_Vis, t)  
      file_results.write(T22_Vis, t)    
      file_results.write(T33_Vis, t) 

'''''''''''''''''''''''
Boundary Conditions
'''''''''''''''''''''''   

# Curve labels from Gmsh
#
# //+
# Physical Curve("left", 5)  
# //+
# Physical Curve("bottom", 6)  
# //+
# Physical Curve("outer", 7)  

# Boundary condition definitions
#bcs_0 = DirichletBC(ME.sub(0).sub(0), 0, facets, 5)  # u1 fix - left  
bcs_1 = DirichletBC(ME.sub(0).sub(1), 0, facets, 6)  # u2 fix - bottom
#
bcs = [bcs_1]

"""
SETUP NONLINEAR PROBLEM
"""
CHProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(CHProblem)

#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-8
prm['newton_solver']['maximum_iterations'] = 30
prm['newton_solver']['error_on_nonconvergence'] = False


print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time 
startTime = datetime.now()

# Give the step a descriptive name
step = "Lithiation"
    
# Write initial state to XDMF file
writeResults(t=0.0)  

# Set increment counter to zero
ii = 0

# initalize output array and counter for time history variables
timeHist0 = np.zeros(shape=[10000])
timeHist1 = np.zeros(shape=[10000])

while (t < Ttot):

    # increment time
    t += float(dk)
        
    # don't allow time to exceed Ttot
    if t>Ttot:
        t -= float(dk) # first, remove too-large dt
        dt = Ttot - t  # then, find dt which makes t=Ttot exactly
        dk.assign(dt)  # assign this new dt  
        t += float(dk) # take the new dt step
    
    # Update time-dependent BCs
    mu_f_exp.t = t

    # Solve the problem
    (iter, converged) = solver.solve()  
         
    # Check if solver converges
    if converged: 
        
        # Print progress of calculation periodically
        if ii%1 == 0:      
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
            print("Simulation Time: {} s | dt: {} s".format(round(t,2), round(dt, 3)))
            print()   
        
        # Iteration-based adaptive time-stepping
        if iter<=2:
            dt = 1.5*dt
            dk.assign(dt)
        elif iter>=5:
            dt = dt/2
            dk.assign(dt)
            
        # Write output to *.xdmf file
        writeResults(t)
        
        # Update DOFs for next step
        w_old.vector()[:] = w.vector()
        
        # Increment counter
        ii += 1
        
        # Store time history variables
        timeHist0[ii] = t # current time
        timeHist1[ii] = assemble(cBar*x[0]*dx)/assemble(x[0]*dx) # state of charge
        

    # If solver doesn't converge, don't save results and try a smaller dt 
    else: 
        # back up in time
        t = t - float(dk)
        
        # cut back on dt
        dt = dt/2
        dk.assign(dt)
        
        # Reset DOFs for next step
        w.vector()[:] = w_old.vector()


'''''''''''''''''''''
       STEP 2
'''''''''''''''''''''

# give the step a descriptive name
step = "Delithiation"

# re-set time step
dt = 0.01
dk.assign(dt)

# expression for mu "step function"
mu_f_exp2 = Expression("t <= tRamp ? mu_f + (mu_i - mu_f)*t/tRamp : mu_i ",
                t = 0.0, tRamp = 1.0, mu_i = mu_i, mu_f=mu_f, degree=1)

''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Equation of motion (test fxn: u)
# Res_1: Balance of mass   (test fxn: mu)
# Res_2: chemical potential (test fxn: c)

# Time step field, constant within body
dk = Constant(dt)

# The weak form for the equation of motion
Res_0 = inner(TR, axi_grad_vector(u_test))*x[0]*dx

# The weak form for the mass balance of mobile species    
Res_1 = dot((c - c_old)/dk, mu_test)*x[0]*dx \
        -  Omega*dot(Jmat , axi_grad_scalar(mu_test) )*x[0]*dx \
        + Omega*dot(mu - mu_f_exp2, mu_test)*x[0]*ds(7)

# The weak form for the concentration
Res_2 = dot(mu - fc/RT, c_test)*x[0]*dx \
        -  dot( (lam/RT)*axi_grad_scalar(cBar), axi_grad_scalar(c_test))*x[0]*dx
        
# Total weak form
Res = Res_0 + Res_1 + Res_2

# Automatic differentiation tangent:
a = derivative(Res, w, dw)

"""
SETUP NONLINEAR PROBLEM
"""
CHProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(CHProblem)

#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-8
prm['newton_solver']['maximum_iterations'] = 30
prm['newton_solver']['error_on_nonconvergence'] = False

while (t < 2*Ttot):

    # increment time
    t += float(dk)
    
    # Update time-dependent BCs
    mu_f_exp2.t = t - Ttot

    # Solve the problem
    (iter, converged) = solver.solve()  
         
    # Check if solver converges
    if converged: 
        
        # Print progress of calculation periodically
        if ii%1 == 0:      
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
            print("Simulation Time: {} s | dt: {} s".format(round(t,2), round(dt, 3)))
            print()   
        
        # Iteration-based adaptive time-stepping
        if iter<=2:
            dt = 1.5*dt
            dk.assign(dt)
        elif iter>=5:
            dt = dt/2
            dk.assign(dt)
            
        # Write output to *.xdmf file
        writeResults(t)
        
        # Update DOFs for next step
        w_old.vector()[:] = w.vector()
        
        # Increment counter
        ii += 1
        
        # Store time history variables
        timeHist0[ii] = t # current time
        timeHist1[ii] = assemble(cBar*x[0]*dx)/assemble(x[0]*dx) # state of charge       

    # If solver doesn't converge, don't save results and try a smaller dt 
    else: 
        # back up in time
        t = t - float(dk)
        
        # cut back on dt
        dt = dt/2
        dk.assign(dt)
        
        # Reset DOFs for next step
        w.vector()[:] = w_old.vector()


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

# set plot font to size 14
font = {'size'   : 14}
plt.rc('font', **font)

# Only plot as far as we have time history data
ind = np.argmax(timeHist0)

# Get array of default plot colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure()
plt.plot(timeHist0[0:ind+1], timeHist1[0:ind+1], linewidth=1.0,\
         color='r', marker='o', markersize=3)
plt.axis('tight')
plt.ylabel(r"State of Charge")
plt.xlabel(r"Time (s)")
plt.grid(linestyle="--", linewidth=0.5, color='b')
plt.ylim(0,1.01)
plt.xlim(0,60)

fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/ellipse_lithiation_step.png", dpi=600)
