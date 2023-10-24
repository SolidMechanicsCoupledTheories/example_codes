"""
Code for Cahn-Hilliard phase separation,
without mechanical coupling.

2D phase separation study.


Degrees of freedom:
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
# Square edge length 
L0 = 0.8 # 800 nm box, after Di Leo et al. (2014) 

# Number of elements along each side
N = 100

# Create square mesh 
mesh = RectangleMesh(Point(0,0), Point(L0, L0), N, N)

# Extract initial mesh coords
x = SpatialCoordinate(mesh) 
    
# Defining periodic boundary conditions in 2-D. 
#
# (See the *.pdf posted on the GitHub repo for more 
#   details on the PeriodicBoundary() function )

class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT
        # on one of the two corners (0, L0) and (L0, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and
                (not ((near(x[0], 0) and near(x[1], L0)) or
                        (near(x[0], L0) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], L0) and near(x[1], L0):
            y[0] = x[0] - L0
            y[1] = x[1] - L0
        elif near(x[0], L0):
            y[0] = x[0] - L0
            y[1] = x[1]
        else:   # near(x[1], L0)
            y[0] = x[0]
            y[1] = x[1] - L0     

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
# Material parameters after Di Leo et al. (2014)
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
t    = 0.0  # initialization of time
Ttot = 2000  # total simulation time 
dt   = 0.01  # Initial time step size, here we will use adaptive time-stepping

'''''''''''''''''''''
Function spaces
'''''''''''''''''''''
# Define function space, both vectorial and scalar
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For  normalized chemical potential
                                                   # and  normalized species concentration
#
TH = MixedElement([P1, P1]) # Mixed element
ME = FunctionSpace(mesh, TH, constrained_domain=PeriodicBoundary())   # Total space for all DOFs

# Define trial functions
w = Function(ME)
mu, c = split(w)  # chemical potential  mu, concentration c

# A copy of functions to store values in the previous step
w_old = Function(ME)
mu_old,  c_old  = split(w_old)   

# Define test functions
w_test = TestFunction(ME)                
mu_test, c_test  = split(w_test)   

# Define trial functions needed for automatic differentiation
dw = TrialFunction(ME)             

# Class for generating the randomized initial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        cBar_rand = 0.63 + 0.05*(0.5 - random.random())
        fc_rand   = float(RT)*(ln(cBar_rand/(1-cBar_rand)) + float(chi)*(1-2*cBar_rand))
        values[1] = float(Omega)*float(cMax)*cBar_rand # Normalized concentration
        values[0] = fc_rand/float(RT) # Normalized chemical potential
    def value_shape(self):
        return (2,)
    
# Create randomized initial chemical potential & concentration conditions,
#  and interpolate.
w_init = InitialConditions(degree=1)
w.interpolate(w_init)
w_old.interpolate(w_init)

'''''''''''''''''''''
SUBROUTINES
'''''''''''''''''''''
'''
For 2D plane strain:
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

#------------------------------------------------------------------------------  
# Species flux
def Flux_calc(mu, c):
    #
    cBar = c/(Omega*cMax) # normalized concentration
    #
    Mob = (D*c)/(Omega*RT)*(1-cBar)
    #
    Jmat = - RT* Mob * pe_grad_scalar(mu)
    return Jmat

# Calculate the f^c term
def fc_calc(mu, c):
    #
    cBar = c/(Omega*cMax) # normalized concentration
    #
    fc   = RT*(ln(cBar/(1-cBar)) + chi*(1-2*cBar) ) 
    #
    return fc

'''''''''''''''''''''''''''''
Kinematics and constitutive relations
'''''''''''''''''''''''''''''

# Calculate the normalized concentration cBar
cBar = c/(Omega*cMax) # normalized concentration

# Calculate the Species flux
Jmat = Flux_calc(mu, c)

# Calculate the f^c term 
fc = fc_calc(mu, c)

''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of mass   (test fxn: mu)
# Res_1: chemical potential (test fxn: c)

# Time step field, constant within body
dk = Constant(dt)

# The weak form for the mass balance of mobile species    
Res_0 = dot((c - c_old)/dk, mu_test)*dx \
        -  Omega*dot(Jmat , pe_grad_scalar(mu_test) )*dx      

# The weak form for the concentration
Res_1 = dot(mu - fc/RT, c_test)*dx \
        -  dot( (lam/RT)*pe_grad_scalar(cBar), pe_grad_scalar(c_test))*dx
        
# Total weak form
Res = Res_0 + Res_1

# Automatic differentiation tangent:
a = derivative(Res, w, dw)
   

'''''''''''''''''''''
 SET UP OUTPUT FILES
'''''''''''''''''''''
# Output file setup
file_results = XDMFFile("results/2D_uncoupled_phase_separation.xdmf")
# "Flush_output" permits reading the output during simulation
# (Although this causes a minor performance hit)
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Function space for projection of results
W = FunctionSpace(mesh,  P1)   # Scalar space for visualization 
     
def writeResults(t):
      # Variable projecting and renaming
      
      # Visualize  the normalized chemical potential
      mu_Vis = project(mu, W)
      mu_Vis.rename("mu"," ")
      
      # Visualize  the normalized concentration
      c_Vis = project(c, W)
      c_Vis.rename("c"," ")
      
      # Visualize cBar
      cBar_Vis = project(cBar, W)
      cBar_Vis.rename("cBar"," ")
   
     # Write field quantities of interest
      file_results.write(mu_Vis, t)
      file_results.write(c_Vis, t)
      file_results.write(cBar_Vis, t)

'''''''''''''''''''''''
Boundary Conditions
'''''''''''''''''''''''   

# Nothing! Just let the system evolve on its own.
bcs = []

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
step = "Evolve"
    
# Write initial state to XDMF file
writeResults(t=0.0)  

# Set increment counter to zero
ii = 0

while (t < Ttot):

    # increment time
    t += float(dk)
    
    # Increment counter
    ii += 1

    # Solve the problem
    (iter, converged) = solver.solve()  
         
    # Now we start the adaptive time-stepping and output storage procedure.
    #
    # First, we check if the newton solver actually converged.
    if converged: 
        
        # If the solver converged, we print the status of the solver, 
        # perform adaptive time-stepping updates, output results, and 
        # update degrees of freedom for the next step, w_old <- w.
        
        # print progress of calculation periodically
        if ii%1 == 0:      
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
            print("Simulation Time: {} s | dt: {} s".format(round(t,2), round(dt, 3)))
            print()   
        
        # Iteration-based adaptive time-stepping
        #
        # If the newton solver takes 2 or less iterations, 
        # increase the time step by a factor of 1.5:
        if iter<=2:
            dt = 1.5*dt
            dk.assign(dt)
        # If the newton solver takes 5 or more iterations, 
        # decrease the time step by a factor of 2:
        elif iter>=5:
            dt = dt/2
            dk.assign(dt)
        # otherwise the newton solver took 3-4 iterations,
        # in which case leave the time step alone.
            
        # Write output to *.xdmf file
        writeResults(t)
        
        # Update DOFs for next step
        w_old.vector()[:] = w.vector()

    # If solver doesn't converge we have to back up in time,
    # cut the size of the time step, and try solving again.
    else: # not(converged)
        
        # first, we back up in time
        # ( to un-do the current time step )
        t = t - float(dk)
        
        # Then, we cut back on the time step we're attempting.
        # (by a factor of 2)
        dt = dt/2
        dk.assign(dt)
        
        # Finally, we have to reset the degrees of freedom to their
        # "old" values before trying to solve again, otherwise the 
        # initial guess for w is retained from the "failed" solve attempt.
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