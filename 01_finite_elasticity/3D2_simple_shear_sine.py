"""
3D Code for  isothermal finite hyperelasticity
Sinusoidal Simple shear

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
# Create mesh and identify the 3D domain and its boundary
#
# Box size, in mm
L= 1.0 
# A basic box mesh 
mesh = BoxMesh(Point(0.,0.,0.),Point(L,L,L),6,6,2)

x = mesh.coordinates()

#Pick up on the boundary entities of the created mesh
class xBot(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0) and on_boundary
    
class xTop(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],1.0) and on_boundary
#
class yBot(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0) and on_boundary
class yTop(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],1.0) and on_boundary
#   
class zBot(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],0) and on_boundary
class zTop(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],1) and on_boundary

# Mark boundary subdomians
facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
DomainBoundary().mark(facets, 7)  # First, mark all boundaries with common index
# Next mark sepcific boundaries
xBot().mark(facets, 1)
xTop().mark(facets, 2)
yBot().mark(facets, 3)
yTop().mark(facets, 4)
zBot().mark(facets, 5)
zTop().mark(facets, 6)
 
# Define surface area measure
ds = Measure("ds", domain=mesh, subdomain_data=facets)

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
# # Initialize time
t = 0
# Cyclical displacement history parameters
gammaAmp = 1.0               # amplitude of  shear strain
uMax     =  L*gammaAmp       # amplitude of displacement. Remember L is the unit box size 
#
ttd     = 2.5                # quarter-cycle time for the sinusoidal input
T_cycle = 4.0*ttd            # cycle time
omega   = 2.* np.pi /T_cycle # frequency in radians per sec
n_cycles = 2                 # Number of cycles
# Total time
Ttot = n_cycles*T_cycle
# Time step
dt = 0.2

# Define expression for displacing the top surface:
dispAmp = Expression(("uMax*sin(omega*t)"),
               omega = omega, uMax = uMax, t = 0.0, degree=1)

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
# Set up output file for visualization
#
file_results = XDMFFile("results/3D_sinusoidal_simple_shear.xdmf")
# "Flush_output" permits reading the output during simulation
# (Although this causes a minor performance hit)
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
        P12_Vis = project(Tmat[0,1],W)
        P12_Vis.rename("P12, kPa","")   
        
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
        file_results.write(P12_Vis, t)  
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
step = "Shear"

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
c = Constant((0.0, 0.0, 0.0))
#
bcs_1 = DirichletBC(ME.sub(0),c, facets, 3)  # u1, u2, u3 fixed - yBot
#
bcs_2 = DirichletBC(ME.sub(0).sub(0), dispAmp, facets, 4)  # u1 move  -  yTop
bcs_3 = DirichletBC(ME.sub(0).sub(1), 0, facets, 4)  # u2 fix    - yTop
bcs_4 = DirichletBC(ME.sub(0).sub(2), 0, facets, 4)  # u3 fixe   - yTop
# BC set
bcs = [bcs_1, bcs_2, bcs_3, bcs_4]

'''''''''''''''''''''''
Define  and solve the nonlinear variational problem
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

# Store start time 
startTime = datetime.now()

# Initalize output array for tip displacement
totSteps = 10000
timeHist0 = np.zeros(shape=[totSteps])
timeHist1 = np.zeros(shape=[totSteps]) 
timeHist2 = np.zeros(shape=[totSteps])
timeHist3 = np.zeros(shape=[totSteps]) 
 
#Iinitialize a counter for reporting data
ii=0

# Time-stepping solution procedure loop
while (round(t + dt, 9) <= Ttot):

    # increment time
    t += dt
    # increment counter
    ii += 1
    
    # update time variable in time-dependent displacement BC
    dispAmp.t = t 

    # Solve the problem
    (iter, converged) = solver.solve()
    
    # Store the results for visualization
    #
    writeResults(t)
    
    # Update DOFs for next step
    w_old.vector()[:] = w.vector()
 
   # Calculate the reaction force at the  boundary where the dispalcement has been prescribed:    
    P        = Tmat_calc(u, p)  
    n        = FacetNormal(ds)
    traction = dot(P,n)
    tangent  = as_vector([1.0, 0.0, 0.0])
    Force    = dot(traction,tangent)*ds(4)

    # Store the displacement and force at this time
    timeHist0[ii] = t # store the current time    
    timeHist1[ii] = gammaAmp*sin(omega*t)  #  Applied displacement
    timeHist2[ii] = assemble(Force)      #  Convert from UFL operator to a number in milli Newtons


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

# set plot font to size 14
font = {'size'   : 14}
plt.rc('font', **font)
# Get array of default plot colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
# Only plot as far as we have time history data
ind = np.argmax(timeHist0)
#
fig = plt.figure() 
#fig.set_size_inches(7,4)
ax=fig.gca()  
plt.plot(timeHist1[:ind], timeHist2[:ind], c='b', linewidth=1.0, marker='.')
#-------------------------------------------------------------
#ax.set.xlim(-0.05,0.05)
#ax.set.ylim(-0.03,0.03)
#plt.axis('tight')
plt.grid(linestyle="--", linewidth=0.5, color='b')
ax.set_xlabel(r'$\gamma$',size=14)
ax.set_ylabel(r'$P_{12}$, kPa ',size=14)
#ax.set_title("Shear stress-strain curve", size=14, weight='normal')
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.show()

fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/3D_finite_elastic_simple_shear.png", dpi=600)
