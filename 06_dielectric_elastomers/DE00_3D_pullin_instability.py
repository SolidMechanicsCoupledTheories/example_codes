"""
3D Code for finite electro-elasticity with u-p formulation.

Problem: Electro-elastic pull-in instability of a 3d VHB  block.

- with basic units:
    > Length: mm
    >   Time:  s
    >   Mass: kg
    > Charge: nC
  and derived units
    > Pressure: kPa 
    > Force: milliNewtons
    > Electric potential: kV
    
Eric M. Stewart    and    Lallit Anand   
(ericstew@mit.edu)        (anand@mit.edu)   

 October 2023
"""

# Fenics-related packages
from dolfin import *
# Numerical array package
import numpy as np
# Plotting packages
from ufl import sinh
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

# The behavior of the form compiler FFC can be adjusted by prescribing
# various parameters. Here, we want to use the UFLACS backend of FFC::
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 4

"""
Create mesh and identify the domain and its boundary
"""
# Bar dimensions
#
L_0  = 1 # mm  Length in x-direction
W_0  = 1 # mm  Width  in y-direction
H_0  = 1 # mm  Height in z-direction
#
length = 1 # mm
#
# Last three numbers below are the number of elements in the three directions
mesh = BoxMesh(Point(0, 0, 0), Point(L_0, W_0, H_0), 3, 3, 3)

# Extract initial mesh coords
x = SpatialCoordinate(mesh)

# Identify the boundary entities of mesh
class face1(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-3
        return near(x[0], 0, tol) and on_boundary
    
class face2(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-3
        return near(x[1], 0, tol) and on_boundary
    
class face3(SubDomain):
    def inside(self, x, on_boundary):
        tol =1E-14
        return near(x[2], 0, tol) and on_boundary
    
    
class face4(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-3
        return near(x[0], L_0, tol) and on_boundary
    
class face5(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-3
        return near(x[1], W_0, tol) and on_boundary
    
class face6(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-3
        return near(x[2], H_0, tol) and on_boundary
    
# Mark boundary subdomains
facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
# First, mark all boundaries with common index
DomainBoundary().mark(facets, 7) 
# Next mark specific boundaries
face1().mark(facets,  1)
face2().mark(facets,  2)
face3().mark(facets,  3)
face4().mark(facets,  4)
face5().mark(facets,  5)
face6().mark(facets,  6)

# Define the boundary integration measure "ds".
ds = Measure('ds', domain=mesh, subdomain_data=facets)

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
# Mechanical parameters
Geq_0   = 15         # Shear modulus, kPa
Kbulk   = 1e3*Geq_0  # Bulk modulus, kPa
I_m     = 175        # Gent locking paramter
# Electrostatic  parameters
vareps_0 = Constant(8.85E-3)         #  permittivity of free space pF/mm
vareps_r = Constant(5)             #  relative permittivity, dimensionless
vareps   = vareps_r*vareps_0         #  permittivity of the material

# Simulation time control-related params
t    = 0.0          # start time (s)
rampRate = 1e-1     # s^{-1}
Ttot = 1.0/rampRate # total simulation time (s) 
numSteps = 100
dt   = Constant(Ttot/numSteps)       # (fixed) step size

# Normalization parameter for voltage: l*sqrt(Geq_0/vareps)
#
phiTot = float(length*np.sqrt(float(Geq_0)/float(vareps)))  # final normalized value of phi

# Boundary condition to ramp up electrostatic potential
phiRamp = Expression(("phi_tot*t/Ttot"), 
                     t = 0.0, phi_tot = phiTot, Ttot=Ttot, degree=1)

'''''''''''''''''''''
FEM SETUP
'''''''''''''''''''''
# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For pressure and electric potential
#
TH = MixedElement([U2, P1, P1])   #  Taylor-Hood style mixed element
ME = FunctionSpace(mesh, TH)      # Total space for all DOFs

# Define actual functions with the required DOFs
w = Function(ME)
u, p, phi = split(w)                   # displacement u, pressure p, potential  phi

# A copy of functions to store values in the previous step for time-stepping
w_old = Function(ME) 
u_old, p_old, phi_old = split(w_old)   # old values

# Define test functions  
w_test = TestFunction(ME)   # Test function
u_test, p_test, phi_test = split(w_test)  # test fields  

#Define trial functions neede for automatic differentiation
dw = TrialFunction(ME)  

'''''''''''''''''''''
SUBROUTINES
'''''''''''''''''''''
# Deformation gradient 
def F_calc(u):
    Id = Identity(3)  
    F  = Id + grad(u)  
    return F

# Generalized shear modulus for Gent model
def Geq_Gent_calc(u):
    F = F_calc(u)
    C = F.T*F
    Cdis = J**(-2/3)*C
    I1 = tr(Cdis)
    z = I1-3 
    z   = conditional( gt(z, I_m), 0.95*I_m, z ) # Keep from blowing up
    Geq_Gent  = Geq_0 * (I_m/(I_m - z))
    return Geq_Gent

# Mechanical Cauchy stress for Gent material
def T_mech_calc(u,p):
    Id = Identity(3) 
    F   = F_calc(u)
    J = det(F)
    B = F*F.T
    Bdis = J**(-2/3)*B
    Geq  = Geq_Gent_calc(u)
    T_mech = (1/J)* Geq * dev(Bdis) - p * Id
    return T_mech

# Maxwell contribution to the Cauchy stress
def T_maxw_calc(u,phi):
    F = F_calc(u)
    e_R  = - grad(phi)    # referential electric field
    e_sp = inv(F.T)*e_R   # spatial electric field 
    # Spatial Maxwel stress
    T_maxw = vareps*(outer(e_sp,e_sp) - 1/2*(inner(e_sp,e_sp))*Identity(3))
    return T_maxw


# Piola  stress
def T_mat_calc(u, p, phi):
    Id = Identity(3) 
    F   = F_calc(u)
    J = det(F)
    #
    T_mech = T_mech_calc(u,p)
    #
    T_maxw = T_maxw_calc(u,phi)
    #
    T      = T_mech + T_maxw
    #
    Tmat   = J * T * inv(F.T)
    return Tmat


# Referential electric displacement 
def Dmat_calc(u, phi):
    F = F_calc(u)
    J = det(F)
    C = F.T*F
    e_R  = - grad(phi) # referential electric field
    Dmat = vareps * J* inv(C)*e_R
    return Dmat

'''''''''''''''''''''''''''''
Kinematics and Constitutive relations
'''''''''''''''''''''''''''''

# Some kinematical quantities
F =  F_calc(u) 
J = det(F)
C =  F.T*F
Fdis = J**(-1/3)*F
Cdis = J**(-2/3)*C

# Mechanical Cauchy stress
T_mech = T_mech_calc(u, p)

# Electrostatic Cauchy stress
T_maxw =T_maxw_calc(u, phi) 

# Piola stress
Tmat = T_mat_calc(u, p, phi)  

# Referential electric displacement
Dmat = Dmat_calc(u, phi)

'''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of forces (test fxn: u)
# Res_1: Pressure variable (test fxn: p)
# Res_2: Gauss's law   (test fxn: phi)

# The weak form for the equilibrium equation
Res_0 =  inner(Tmat, grad(u_test) )*dx

# The weak form for the pressure  
Res_1 =  dot((p/Kbulk + ln(J)/J) , p_test)*dx

#  The weak form for Gauss's equation
Res_2 = inner(Dmat, grad(phi_test))*dx 

# Total weak form
Res  =  Res_0 + Res_1 + Res_2

# Automatic differentiation tangent:
a = derivative(Res, w, dw)
   
'''''''''''''''''''''
 SET UP OUTPUT FILES
'''''''''''''''''''''
# Output file setup
file_results = XDMFFile("results/3D_pullin_Im175.xdmf")
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
        
        # Visualize  normalized chemical potential
        phi_Vis = project(phi,W)
        phi_Vis.rename("phi, kV"," ")   
        
        
        # Visualize J
        J_Vis = project(J, W)
        J_Vis.rename("J"," ")    
    
        
        # Visualize the Mises stress  
        T    = Tmat*F.T/J
        T0   = T - (1/3)*tr(T)*Identity(3)
        #
        Mises = sqrt((3/2)*inner(T0, T0))
        Mises_Vis = project(Mises,W)
        Mises_Vis.rename("Mises"," ")    
        

        T11_Mech_Vis = project(T_mech[0,0],W)
        T11_Mech_Vis.rename("T11_Mech, kPa","")
        T22_Mech_Vis = project(T_mech[1,1],W)
        T22_Mech_Vis.rename("T22_Mech, kPa","")  
        T33_Mech_Vis = project(T_mech[2,2],W)
        T33_Mech_Vis.rename("T33_Mech, kPa","")  
        
        T11_Maxw_Vis = project(T_maxw[0,0],W)
        T11_Maxw_Vis.rename("T11_Maxw, kPa","")
        T22_Maxw_Vis = project(T_maxw[1,1],W)
        T22_Maxw_Vis.rename("T22_Maxw, kPa","")  
        T33_Maxw_Vis = project(T_maxw[2,2],W)
        T33_Maxw_Vis.rename("T33Maxw, kPa","") 
        
        T11_Vis = project(T[0,0],W)
        T11_Vis.rename("T11, kPa","")
        T22_Vis = project(T[1,1],W)
        T22_Vis.rename("T22, kPa","")   
        T33_Vis = project(T[2,2],W)
        T33_Vis.rename("T33, kPa","")     
 
       # Write field quantities of interest
        file_results.write(u_Vis, t)
        file_results.write(p_Vis, t)
        file_results.write(phi_Vis, t)
        file_results.write(J_Vis, t)  
        file_results.write(Mises_Vis, t)
        file_results.write(T11_Mech_Vis, t)  
        file_results.write(T22_Mech_Vis, t)  
        file_results.write(T33_Mech_Vis, t)          
        file_results.write(T11_Maxw_Vis, t)  
        file_results.write(T22_Maxw_Vis, t)   
        file_results.write(T33_Maxw_Vis, t)          
        file_results.write(T11_Vis, t)  
        file_results.write(T22_Vis, t)   
        file_results.write(T33_Vis, t)                 
        
# Write initial state to XDMF file
writeResults(t=0.0)       


print("------------------------------------")
print("Start Simulation")
print("------------------------------------")
# Store start time 
startTime = datetime.now()

"""""""""""""""""
     STEP
"""""""""""""""""
# Give the step a descriptive name
step = "Actuate"

'''''''''''''''''''''''
Boundary conditions
'''''''''''''''''''''''



bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 1)    # fix u1 degree  on face1 
bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 2)    # fix u2 dgree   on face2
bcs_3 = DirichletBC(ME.sub(0).sub(2), 0, facets, 3)    # fix u3 degree  on face3
#
bcs_4 = DirichletBC(ME.sub(2), 0, facets, 3)           #  phi ground bottom, face 3
bcs_5 = DirichletBC(ME.sub(2), phiRamp, facets, 6)     #  phi ramp top, face 6 
#

# BC sets for different steps of simulation
bcs = [bcs_1, bcs_2, bcs_3, bcs_4, bcs_5]


# Set up the non-linear problem 
electrostaticProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
 
# Set up the non-linear solver
solver  = NonlinearVariationalSolver(electrostaticProblem)

#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance']   = 1.e-8
prm['newton_solver']['relative_tolerance']   = 1.e-7
prm['newton_solver']['maximum_iterations']   = 25
prm['newton_solver']['relaxation_parameter'] = 1.0
#prm['newton_solver']['error_on_nonconvergence'] = False

# Initalize output array for tip displacement
totSteps = numSteps+1
timeHist0 = np.zeros(shape=[totSteps])
timeHist1 = np.zeros(shape=[totSteps]) 
#Iinitialize a counter for reporting data
ii=0

# Time-stepping solution procedure loop
while (round(t + dt, 9) <= Ttot):
    
    # increment time
    t += float(dt)
    # increment counter
    ii += 1
    
    # update time variables in time-dependent BCs 
    phiRamp.t = t
    
    # Solve the problem
    try:
        (iter, converged) = solver.solve()
    except: # Break the loop if solver fails
        break
    
    # Write output to *.xdmf file
    writeResults(t)
    
    # Update DOFs for next step
    w_old.vector()[:] = w.vector()

    
    # Store  displacement and potential at a particular point  at this time
    timeHist0[ii] = w.sub(0).sub(2)(length, length, length) # time history of displacement
    timeHist1[ii] = w.sub(2)(length, length, length)        # time history of voltage phi

    # print progress of calculation 
    if ii%1 == 0:      
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
        print("Simulation Time: {} s | dt: {} s".format(round(t,2), round(dt, 3)))
        print()   
            


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

# Get array of default plot colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Plot the normalized dimensionless quantity for $\phi$ used in Wang et al. 2016
# versus stretch in the vertical direction.
#
normVolts = timeHist1/(length * np.sqrt(float(Geq_0)/float(vareps))) 
normVolts = normVolts[0:ii]
#
stretch = timeHist0/length + 1.0
stretch = stretch[0:ii]
#
plt.plot(normVolts, stretch, c=colors[0], linewidth=1.0, marker='.')
#plt.scatter(normVolts[ii-1], stretch[ii-1], c='k', marker='x', s=100)
#plt.scatter(normVolts[ii-1], stretch[ii-1], c='k',  s=100)
plt.grid(linestyle="--", linewidth=0.5, color='b')
ax = plt.gca()
#
ax.set_ylabel(r'$\lambda$')
ax.set_ylim([0.0,1.1])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
#
ax.set_xlabel(r'$(\phi/\ell_0)/\sqrt{G_0/\varepsilon} $')
ax.set_xlim([0.2,1.0])
ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
#
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.show()

fig = plt.gcf()
fig.set_size_inches(6,4)
plt.tight_layout()
plt.savefig("results/cube_pullin_Im175.png", dpi=600)
