"""
Code for axisymmetric  coupled thermoelasticity of elastomers

Contraction of a stretched rubber cylinder upon heating
This axi-symmetric simulation is performed  in two steps:
    
Step 1: Subject the cylinder to  a traction ramped from 0 to 2MPa
        on one of its end faces, while maintianing the
        temperature constant at 298K on its exterior faces
        
Step 2: While keeping the end traction constant, ramp
        up the boundary temperature from 298 to 348K
        

Degrees of freedom:
#
vectorial displacement: u
pressure: p
temperature: theta
    

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
Boltzmann Constant: 1.38E-17 microJ/K
Number of polymer chains per unit vol: #/mm^3
Thermal expansion coefficient: #/K
Specific heat: microJ/(mm^3 K)
Thermal conductivity: microW/(mm K)

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
# Create 2D mesh for an axi-symmetric simulation
R0 = 10.0  # mm radius of cylinder
H0 = 10.0  # mm height of cylinder
# Last two numbers below are the number of elements in the two directions
mesh = RectangleMesh(Point(0, 0), Point(R0, H0), 20, 20, "crossed")

x = SpatialCoordinate(mesh)

# Identify the boundaries of the  mesh
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
k_B      = Constant(1.38E-17)             # Boltzmann's constant
theta0   = Constant(298)                  # Initial temperature
Gshear_0 = Constant(280)                  # Ground sate shear modulus
N_R      = Constant(Gshear_0/(k_B*theta0))# Number polymer chains per unit ref. volume
lambdaL  = Constant(5.12)                 # Locking stretch
Kbulk    = Constant(1000*Gshear_0)        # Bulk modulus
alpha    = Constant(180.0E-6)             # Coefficient of thermal expansion
c_v      = Constant(1930)                 # Specific heat
k_therm  = Constant(0.16E3)               # Thermal conductivity

"""
Simulation time-control related params
"""
t          = 0.0       
step1_time = 50   
# 
step2_time = 250
step2_ramp = 50  # In Step 2 the  temperature is ramped up by 50K 
                 # over 50 seconds and then held constant for 200 seconds 
#
Ttot = step1_time + step2_time  
dt   = 2           # Fixed  time step size

'''''''''''''''''''''
Function spaces
'''''''''''''''''''''
# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For  pressure and temperature
                                                 
#
TH = MixedElement([U2, P1, P1]) # Taylor-Hood style mixed element
ME = FunctionSpace(mesh, TH)    # Total space for all DOFs

# Define actual functions with the required DOFs
w = Function(ME)
u, p,theta = split(w)  # displacement u, pressure p, temperature theta

# A copy of functions to store values in the previous step for time-stepping
w_old = Function(ME)
u_old,  p_old, theta_old = split(w_old)   

# Define test functions  
w_test = TestFunction(ME)                
u_test, p_test,  theta_test = split(w_test)   

# Define trial functions needed for automatic differentiation
dw = TrialFunction(ME)                  

# Assign initial  temperature  theta0 to the body
init_theta = Constant(theta0)
theta_init = interpolate(init_theta, ME.sub(2).collapse())
assign(w_old.sub(2), theta_init)
assign(w.sub(2), theta_init)

'''''''''''''''''''''
SUBROUTINES
'''''''''''''''''''''
'''
For axisymmetric:
'''
# Special gradient operators for axisymmetric functions 
#
# Gradient of vector field u   
def ax_grad_vector(u):
    grad_u = grad(u)
    return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, u[0]/x[0]]]) 

# Gradient of scalar field y
# (just need an extra zero for dimensions to work out)
def ax_grad_scalar(y):
    grad_y = grad(y)
    return as_vector([grad_y[0], grad_y[1], 0.])

# Axisymmetric deformation gradient 
def F_ax_calc(u):
    dim = len(u)
    Id = Identity(dim)          # Identity tensor
    F = Id + grad(u)            # 2D Deformation gradient
    F33_exp =  (x[0]+u[0])/x[0]  # axisymmetric F33, R/R0 
    F33 = conditional(lt(x[0], DOLFIN_EPS), 1.0, F33_exp) # avoid divide by zero at r=0  
    return as_tensor([[F[0,0], F[0,1], 0],
                  [F[1,0], F[1,1], 0],
                  [0, 0, F33]]) # Full axisymmetric F

def lambdaBar_calc(u):
    F    = F_ax_calc(u)
    J    = det(F)
    C    = F.T*F
    Cdis = J**(-2/3)*C
    I1   = tr(Cdis)
    lambdaBar = sqrt(I1/3.0)
    return lambdaBar

def zeta_calc(u):
    lambdaBar = lambdaBar_calc(u)
    # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
    z    = lambdaBar/lambdaL
    z    = conditional(gt(z,0.95), 0.95, z) # Prevent the simulation from blowing up
    beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
    zeta = (lambdaL/(3*lambdaBar))*beta
    return zeta

# Piola stress 
def Tmat_calc(u, p, theta):
    F = F_ax_calc(u)
    J = det(F)
    #
    zeta = zeta_calc(u)
    Gshear  = N_R * k_B * theta * zeta
    #
    Tmat = J**(-2/3) * Gshear * (F - (1/3)*tr(C)*inv(F.T) ) - J * p * inv(F.T) 
    return Tmat

# Calculate the stress temperature tensor
def M_calc(u):
    Id  = Identity(3)         
    F   = F_ax_calc(u) 
    #
    C  = F.T*F
    Cinv = inv(C) 
    J = det(F)
    zeta = zeta_calc(u)
    #
    fac1 = N_R * k_B * zeta
    fac2 = (3*Kbulk*alpha)/J
    #
    M =  J**(-2/3) * fac1 * (Id - (1/3)*tr(C)*Cinv)  - J * fac2 * Cinv
    return M
    
#  Heat flux
def Heat_flux_calc(u, theta):
    F = F_ax_calc(u) 
    J = det(F)         
    #
    Cinv = inv(F.T*F) 
    #
    Tcond = J * k_therm * Cinv # Thermal conductivity tensor
    #
    Qmat = - Tcond * ax_grad_scalar(theta)
    return Qmat


'''''''''''''''''''''''''''''
Evaluate kinematics and constitutive relations
'''''''''''''''''''''''''''''

# Kinematics
F = F_ax_calc(u)
J = det(F)   
#
lambdaBar = lambdaBar_calc(u)
#
F_old = F_ax_calc(u_old)
J_old = det(F_old)   
#
C     = F.T*F
C_old = F_old.T*F_old

#  Piola stress
Tmat = Tmat_calc(u, p, theta)

# Calculate the stress-temperature tensor
M = M_calc(u)

# Calculate the heat flux
Qmat = Heat_flux_calc(u,theta)

''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of forces (test fxn: u)
# Res_1: Coupling pressure (test fxn: p)
# Res_2: Balance of energy  (test fxn: thetau)

# Expression for applied traction
trac_max = 2000 # kPa = 2MPa
tracRamp = Expression("min(trac_max*t/tRamp, trac_max)", \
              trac_max=trac_max, t= 0.0, tRamp =step1_time, degree=1)
traction = as_vector([0, tracRamp])

# Time step field, constant within body
dk = Constant(dt)

# The weak form for the equilibrium equation
Res_0 = inner(Tmat, ax_grad_vector(u_test))*x[0]*dx\
        - dot(traction, u_test)*x[0]*ds(4)

# The weak form for the pressure
fac_p =  ( ln(J) - 3*alpha*(theta-theta0) )/J
#
Res_1 = dot( (p/Kbulk + fac_p), p_test)*x[0]*dx
      

# The weak form for heat equation
Res_2 = dot( c_v*(theta - theta_old), theta_test)*x[0]*dx \
        -  (1/2)*theta * inner(M, (C - C_old)) * theta_test*x[0]*dx \
        -  dk* dot(Qmat , ax_grad_scalar(theta_test) )*x[0]*dx   
        
# Total weak form
Res = Res_0 +  Res_1 + Res_2 

# Automatic differentiation tangent:
a = derivative(Res, w, dw)

'''''''''''''''''''''
 SET UP OUTPUT FILES
'''''''''''''''''''''
# Output file setup
file_results = XDMFFile("results/Axi_thermoelas_steps.xdmf")
# "Flush_output" permits reading the output during simulation
# (Although this causes a minor performance hit)
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True


# Function space for projection of results
W2 = FunctionSpace(mesh, U2) # Vector space for visualization  
W = FunctionSpace(mesh,P1)   # Scalar space for visualization 

def writeResults(t):
    
    # Variable projecting and renaming
     u_Vis = project(u, W2)
     u_Vis.rename("disp"," ")
     
     # Visualize the pressure
     p_Vis = project(p, W)
     p_Vis.rename("p"," ")
     
     # Visualize  the temperature
     theta_Vis = project(theta, W)
     theta_Vis.rename("theta"," ")
     
     # Visualize J
     J_Vis = project(J, W)
     J_Vis.rename("J"," ")    
 
     # Visualize effective stretch
     lambdaBar_Vis = project(lambdaBar,W)
     lambdaBar_Vis.rename("LambdaBar"," ")
     
     # Visualize M:Cdot
     MCdot = inner(M, (C-C_old)/dt)
     MCdot_Vis = project(MCdot, W)
     MCdot_Vis.rename("M:Cdot", "")
     
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
     file_results.write(theta_Vis, t)
     #
     file_results.write(J_Vis, t)  
     file_results.write(lambdaBar_Vis, t)
     #
     file_results.write(P11_Vis, t)  
     file_results.write(P22_Vis, t)    
     file_results.write(P33_Vis, t)  
     file_results.write(Mises_Vis, t)
     file_results.write(MCdot_Vis, t)              
        
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
     STEP 1
"""""""""""""""""
# Give the step a descriptive name
step = "Step 1"

"""
DIRCHELET BOUNDARY CONDITIONS 
"""
bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 1) # u1 fix - Left
bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 2) # u2 fix - Bottom
#
bcs_3 = DirichletBC(ME.sub(2), theta0, facets, 2)  # Temperature - Bottom
bcs_4 = DirichletBC(ME.sub(2), theta0, facets, 3)  # Temperature - Right
bcs_5 = DirichletBC(ME.sub(2), theta0, facets, 4)  # Temperature - Top
# 
bcs = [bcs_1, bcs_2, bcs_3, bcs_4, bcs_5]

"""
SETUP NONLINEAR PROBLEM
"""
ElasProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(ElasProblem)
#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-8
prm['newton_solver']['maximum_iterations'] = 30

# Initialize an  array for storing results
siz       = 100000 
Forc      = np.zeros([siz+1,4])
Forc[0,3] = theta0    # initial temp

#Iinitialize a counter for reporting data
ii=0

while (t < step1_time):
    
    # increment time
    t += dt
    
    # increment counter 
    ii+=1   

    # update time variables in time-dependent BCs
    tracRamp.t = t

    # Solve the problem
    (iter, converged) = solver.solve()  
    
    # Store the results for visualization
    writeResults(t)
 
    # Update DOFs for next step
    w_old.vector()[:] = w.vector()
    
    # Write time histories 
    #
    Forc[ii,0] = w.sub(0).sub(1)(0, H0) # displacement of a point on the top-face   
    #
    Tmat = Tmat_calc(u, p, theta)
    P22  = project(Tmat[1,1], W)
    Forc[ii,1] = assemble(2*np.pi*P22*x[0]*ds(4)) # Reaction force on the top face
    #
    Forc[ii,2] = t # store the time 
    #
    Forc[ii,3] = w.sub(2)(0, H0) # store surface temperature

    # Print progress of calculation
    if ii%1 == 0:      
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: {} | Simulation Time: {} s, Wallclock Time: {}".\
              format(step, round(t,4), current_time))
        print("Iterations: {}".format(iter))
        print()  
        
"""""""""""""""""
      STEP 2
"""""""""""""""""
      
# Give the second step a descriptive name
step = "Step 2"
   
"""
DIRICHILET BOUNDARY CONDITIONS 
"""

thetaRamp = Expression("t >= step1_time ? \
                       min(theta0 + deltaTheta*(t-step1_time)/tRamp, theta0 + deltaTheta):theta0", \
            theta0 = 298, deltaTheta = 50, t= 0.0, step1_time=step1_time, tRamp = step2_ramp, degree=1)

# Boundary condition definitions
# bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 1) # u1 fix - Left
bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 2) # u2 fix - Bottom
#
#bcs_3 = DirichletBC(ME.sub(2), thetaRamp, facets, 1)  # Temperature - Left
bcs_3 = DirichletBC(ME.sub(2), thetaRamp, facets, 2)  # Temperature - Bottom
bcs_4 = DirichletBC(ME.sub(2), thetaRamp, facets, 3)  # Temperature - Right
bcs_5 = DirichletBC(ME.sub(2), thetaRamp, facets, 4)  # Temperature - Top

# 
bcs = [bcs_1, bcs_2, bcs_3, bcs_4, bcs_5]

"""
SETUP NONLINEAR PROBLEM
"""
ElasProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(ElasProblem)
#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-8
prm['newton_solver']['maximum_iterations'] = 30


while (t < step1_time + step2_time):
    
    # increment time
    t += dt
    
    # increment counter 
    ii+=1   
    
    # update time variables in time-dependent BCs
    tracRamp.t = t
    thetaRamp.t = t 

    # Solve the problem
    (iter, converged) = solver.solve()  
    
    # Store the results for visualization
    writeResults(t)
 
    # Update DOFs for next step
    w_old.vector()[:] = w.vector()
    
    # Write time histories 
    #
    Forc[ii,0] = w.sub(0).sub(1)(0, H0) # displacement of a point on the top-face   
    #
    Tmat = Tmat_calc(u, p, theta)
    P22  = project(Tmat[1,1], W)
    Forc[ii,1] = assemble(2*np.pi*P22*x[0]*ds(4)) # Reaction force on the top face
    #
    Forc[ii,2] = t # store the time 
    #
    Forc[ii,3] = w.sub(2)(0, H0) # store surface temperature

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
ind = np.argmax(Forc[:,2])
   
# Create figure for stress-stetch curve.
#
fig = plt.figure() 
ax=fig.gca()  
plt.plot( Forc[0:ind,1]/(pi*R0*R0)/1000, (Forc[0:ind,0]/H0 +1), c='b', linewidth=1.0, marker='.')
#-----------------------------------------------------
plt.grid(linestyle="--", linewidth=0.5, color='b')
ax.set_xlabel("Stress, MPa",size=14)
ax.set_ylabel("Stretch",size=14)
ax.set_title("Stress-stretch curve", size=14, weight='normal')
#
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
#plt.legend()
import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
plt.show()
# Save figure to file
fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/axi_stress_stretch.png", dpi=600) 


# Create figure for temperature-stretch curve.
#
fig = plt.figure() 
ax=fig.gca()  
plt.plot( Forc[0:ind,3], (Forc[0:ind,0]/H0 +1), c='r', linewidth=1.0, marker='.')
#----------------------------------------------------------------------------
plt.grid(linestyle="--", linewidth=0.5, color='b')
ax.set_xlabel("Surface Temperature, K",size=14)
ax.set_ylabel("Stretch",size=14)
ax.set_title("Temperature-stretch curve", size=14, weight='normal')
#
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
#plt.legend()
import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
plt.show()
# Save figure to file
fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/axi_temp_stretch.png", dpi=600) 
