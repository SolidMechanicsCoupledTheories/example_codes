"""
3D Code for  isothermal finite hyperelasticity

Simple stretch

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
# A 3-D cube
length = 10.0 # mm
mesh = BoxMesh(Point(0.,0.,0.),Point(length,length,length),2,2,2)

# Extract initial mesh coords
x = SpatialCoordinate(mesh)

# Identify the planar boundaries of the  box mesh
#
class xBot(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0) and on_boundary
    
class xTop(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],length) and on_boundary
    
class yBot(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0.0) and on_boundary
    
class yTop(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],length) and on_boundary
    
class zBot(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],0.0) and on_boundary

    
class zTop(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],length) and on_boundary
    

# Mark boundary subdomains
facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
DomainBoundary().mark(facets, 7)  # First, mark all boundaries with common index
# Next mark specific boundaries
xBot().mark(facets, 1)
xTop().mark(facets, 2)

yBot().mark(facets, 3)
yTop().mark(facets, 4)

zBot().mark(facets, 5)
zTop().mark(facets, 6)

# Define the boundary integration measure "ds"
ds = Measure('ds', domain=mesh, subdomain_data=facets)
#  Define facet normal
n = FacetNormal(mesh)

'''''''''''''''''''''
MATERIAL PARAMETERS
Arruda-Boyce Model
'''''''''''''''''''''
Gshear_0 = Constant(280.0)            # Ground state shear modulus
lambdaL  = Constant(5.12)             # Locking stretch
Kbulk    = Constant(1000.0*Gshear_0)  # Bulk modulus

"""
Simulation time-control related params
"""
t        = 0.0      # start time (s)
stretch  = 7.75      # stretch amplitude
dispTot  = (stretch-1)*length
rate     = 1.e-1
Ttot     = (stretch-1)/rate
numSteps = 100
dt       = Ttot/numSteps  # (fixed) step size

# Define expression to ramp up displacement boundary condition
dispRamp = Expression(("dispTot*t/Ttot"),
                    t = 0.0, dispTot = dispTot, Ttot=Ttot, degree=1)

'''''''''''''''''''''
Function spaces
'''''''''''''''''''''
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
def Piola_calc(u, p):
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
Tmat = Piola_calc(u, p)

''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of forces (test fxn: u)
# Res_1: Coupling pressure (test fxn: p)

# The weak form for the equilibrium equation. No body force
Res_0 = inner(Tmat , grad(u_test) )*dx

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
# Output file setup
file_results = XDMFFile("results/3D_finite_elastic_stretch.xdmf")
# "Flush_output" permits reading the output during simulation
#
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
        Mises_Vis.rename("Mises"," ")    

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
step = "Stretch"

"""
BOUNDARY CONDITIONS 
"""
# Recall the markers for the facets of the cube
#
# xBot().mark(facets, 1)
# xTop().mark(facets, 2)
# yBot().mark(facets, 3)
# yTop().mark(facets, 4)
# zBot().mark(facets, 5)
# zTop().mark(facets, 6)

# Boundary condition definitions
#
bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 1)  # u1 fix - xBot
bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 3)  # u2 fix - yBot
bcs_3 = DirichletBC(ME.sub(0).sub(2), 0, facets, 5)  # u3 fix - zBot
#
bcs_4 = DirichletBC(ME.sub(0).sub(1), dispRamp, facets, 4)  # disp ramp - yTop

# BC set 
bcs = [bcs_1, bcs_2, bcs_3, bcs_4]

'''''''''''''''''''''''
Define the nonlinear variational problem
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

# Initialize time history variables.
totSteps = numSteps+1
timeHist0 = np.zeros(shape=[totSteps])
timeHist1 = np.zeros(shape=[totSteps]) 

#Iinitialize a counter for reporting data
ii=0

# Time-stepping solution procedure loop
while (round(t + dt, 9) <= Ttot):
     
    # increment time
    t += dt 
    # increment counter
    ii += 1
    
    # update time variables in time-dependent BCs 
    dispRamp.t = t
    
    # Solve the problem
    try:
        (iter, converged) = solver.solve()
    except: # Break the loop if solver fails
        break
    
    # Write output to *.xdmf file
    writeResults(t)
    
    # Update DOFs for next step
    w_old.vector()[:] = w.vector()
   
    # Store  displacement at a particular point  at this time
    timeHist0[ii] = w.sub(0).sub(1)(length, length, length) # time history of displacement
    #
    P22 = project(Tmat[1,1],W)
    timeHist1[ii] = assemble(P22*ds(4))/(length*length)  # time history of engineering stress
    
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

#plt.figure()
plt.plot((length + timeHist0)/length, timeHist1/1.E3, linewidth=2.0,\
         color=colors[0], marker='.')
plt.axis('tight')
plt.ylabel(r'$P_{22}$, MPa')
plt.xlabel(r'$\lambda$')
plt.xlim([1,8])
plt.ylim([0,8])
plt.grid(linestyle="--", linewidth=0.5, color='b')
# plt.show()   

fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/3D_finite_elastic_stress_stretch.png", dpi=600)
 
