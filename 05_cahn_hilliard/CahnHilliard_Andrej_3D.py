"""
FEniCS code for 3D phase-separation using the Cahn-Hilliard theory.

Degrees of freedom: 
    > c : concentration
    > mu: chemical potential

Adapted from an example problem in Andrej Kosmrlj's very nice FEniCS tutorial, 
https://github.com/akosmrlj/FEniCS_tutorial/blob/master/CahnHilliard/CahnHilliard.ipynb

Lallit Anand   and Eric Stewart
anand@mit.edu     ericstew@mit.edu

September 2023
"""

from dolfin import *
from mshr import *
import numpy as np
import random
import matplotlib.pyplot as plt
plt.close('all') # clear all plots from previous simulation
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
N=24 # number of elements along each side
mesh = BoxMesh(Point(0,0,0), Point(1,1,1), N, N, N)

"""
Periodic boundary conditions in 3-D are significantly more complex. 

See the *.pdf posted on the GitHub repo for more details.
"""

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on x,y,z=0 boundary AND NOT
        # on one of the six edges at the interface between the two regions.
        def is_edge(x):
            return  bool((near(x[0], 1) and near(x[2], 0)) or
                         (near(x[1], 1) and near(x[2], 0)) or
                         (near(x[1], 1) and near(x[0], 0)) or
                         (near(x[2], 1) and near(x[0], 0)) or
                         (near(x[2], 1) and near(x[1], 0)) or
                         (near(x[0], 1) and near(x[1], 0)) )
        #
        return bool((near(x[0], 0) or near(x[1], 0) or near(x[2], 0)) and
                (not ( is_edge(x) ) and on_boundary))

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1) and near(x[2], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
            y[2] = x[2] - 1.
        elif near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.
            y[1] = x[1] - 1.
            y[2] = x[2]
        elif near(x[0], 1) and near(x[2], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
            y[2] = x[2] - 1.
        elif near(x[1], 1) and near(x[2], 1):
            y[0] = x[0]
            y[1] = x[1] - 1.
            y[2] = x[2] - 1.
        elif near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
            y[2] = x[2]
        elif near(x[1], 1):
            y[0] = x[0]
            y[1] = x[1] - 1.
            y[2] = x[2]
        else:   # near(x[2], 1)
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - 1.


"""
Simulation time-control related params
"""
# Step in time
t         = 0.0
dt        = 5.0e-06  # time step size
num_steps = 10
T         = num_steps*dt

'''''''''''''''''''''
Function spaces
'''''''''''''''''''''

# Define function space with periodic boundary conditions
P = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
MFS = FunctionSpace(mesh, MixedElement([P,P]),constrained_domain=PeriodicBoundary())

# Define functions
w = Function(MFS)   # solution for the next step
w_old  = Function(MFS)  # solution from previous step

# Split mixed functions
c, mu  = split(w)
c_old, mu_old  = split(w_old)

# Define test functions
tf = TestFunction(MFS)
c_test, mu_test  = split(tf)

# Define trial  functions for calculating the Jacobian
dw = TrialFunction(MFS)
 

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.63 + 0.02*(0.5 - random.random()) # concentration
        values[1] = 0.0 # chemical potential
    def value_shape(self):
        return (2,)
    

# Create intial conditions and interpolate
w_init = InitialConditions(degree=1)
w.interpolate(w_init)
w_old.interpolate(w_init)

''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''

# Compute the chemical potential df/dc
c = variable(c)
f    = 100*c**2*(1-c)**2
dfdc = diff(f, c)

# Material parameter
lmbda  = 1.0e-02  # surface parameter

# Time-stepping parameter
dt     = 5.0e-06  # time step
theta  = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# mu_(n+theta)
mu_mid = (1.0-theta)*mu_old + theta*mu

# Weak Forms
Res_0 = (c - c_old)/dt*c_test*dx + dot(grad(mu_mid), grad(c_test))*dx
Res_1 = mu*mu_test*dx - dfdc*mu_test*dx - lmbda*dot(grad(c), grad(mu_test))*dx
Res = Res_0 + Res_1

# Compute directional derivative about w in the direction of dw (Jacobian) 
J = derivative(Res, w, dw)

"""
SETUP NONLINEAR PROBLEM
"""

# Set up the non-linear solver
problem = NonlinearVariationalProblem(Res,w,[],J)
solver  = NonlinearVariationalSolver(problem)

# Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'mumps'
prm['newton_solver']['absolute_tolerance'] = 1E-10
prm['newton_solver']['relative_tolerance'] = 1E-9
prm['newton_solver']['maximum_iterations'] = 16

# Note that we here used the Newton-Raphson method,
# which can be used to solve both linear and non-linear problems. 
# Consult with the FEniCS documentation for other solver options.


# Setup  file for output of results for paraview
file_results = XDMFFile("results/cahn_hilliard_3D.xdmf")
# "Flush_output" permits reading the output during simulation
# (Although this causes a minor performance hit)
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True 

#
print("------------------------------------")
print("Start computation")
print("------------------------------------")

# Store start time 
startTime = datetime.now()
# Give the step a descriptive name
step = "Evolve"
# initialize a step counter
ii = 0


while (t < T):
    ii+= 1
    t += dt
    #
    w_old.vector()[:] = w.vector()
    
    (iter, converged) = solver.solve()
    
    (c, mu) = w.split()
    
    # write results for Paraview visualization
    c.rename('c','') 
    file_results.write(c,t)

    mu.rename('mu','') 
    file_results.write(mu,t)
     
    # Print progress of calculation
    if ii%1 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: {} |   Simulation Time: {}  s  |     Wallclock Time: {}".format(step, round(t,9), current_time))
        print("Iterations: {}".format(iter))
        print()
 
# End analysis
print("------------------------------------")
print("End computation") 
# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("Elapsed time:  {}".format(elapseTime))
print("------------------------------------")
