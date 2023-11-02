"""
3D Code for finite electro-elasticity.

Problem: Electro-elastic curling of a sheet.

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
# A 3-D rectangular membrane 
length = 100. # mm
width  = 50.  # mm
thickness = 1 # mm
mesh = BoxMesh(Point(0.,0.,0.),Point(length,thickness, width),25,4,8)

x = mesh.coordinates()

#Pick up on the boundary entities of the created mesh
class Left(SubDomain):
    def inside(self, x, on_boundary):
        tol = 0.1
        return near(x[0],0, tol) and on_boundary
    
class LeftBtm(SubDomain):
    def inside(self, x, on_boundary):
        tol = 0.1
        return near(x[0],0, tol) and x[1]<=thickness/2 and on_boundary
    
class Right(SubDomain):
    def inside(self, x, on_boundary):
        tol = 0.1
        return near(x[0],length, tol) and on_boundary
    
class Top(SubDomain):
    def inside(self, x, on_boundary):
        tol = 0.1
        return near(x[1],thickness, tol) and on_boundary
    
    
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        tol = 0.1
        return near(x[1],0.0, tol) and on_boundary
    
class Center(SubDomain):
    def inside(self, x, on_boundary):
        tol = 0.1
        return near(x[1],thickness/2.0, tol)
    
class Front(SubDomain):
    def inside(self, x, on_boundary):
        tol = 0.1
        return near(x[2],width, tol) and on_boundary
    
class Back(SubDomain):
    def inside(self, x, on_boundary):
        tol = 0.1
        return near(x[2],0.0, tol) and on_boundary

# Mark boundary subdomains
facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
DomainBoundary().mark(facets, 1)  # First, mark all boundaries with common index
# Next mark specific boundaries
Left().mark(facets, 2)
LeftBtm().mark(facets, 3)
Right().mark(facets,4)
Top().mark(facets,5)
Bottom().mark(facets,6)
Center().mark(facets,7)
Front().mark(facets, 8)
Back().mark(facets, 9)

# Define the boundary integration measure "ds".
ds = Measure('ds', domain=mesh, subdomain_data=facets)


'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''

# Mechanical parameters
Geq_0   = 34     # Shear modulus, kPa
Kbulk   = 1e3*Geq_0  # Bulk modulus, kPa
I_m     = 90        # Gent locking paramter
# Electrostatic  parameters
vareps_0 = Constant(8.85E-3)         #  permittivity of free space pF/mm
vareps_r = Constant(6.5)            #  relative permittivity, dimensionless
vareps   = vareps_r*vareps_0         #  permittivity ofthe material


# Simulation time control-related params
t        = 0.0          # start time (s)
Ttot     = 2.0          # total simulation time (s) 
phiTot   = 4.0 # kV
numSteps = 50
dt       = Ttot/numSteps       # (fixed) step size
dk       = Constant(dt)

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
u, p, phi = split(w)                   # dispalacement u, pressure p, potential, phi

# A copy of functions to store values in the previous step for time-stepping
w_old = Function(ME) 
u_old, p_old, phi_old = split(w_old)   # old values

# Define test functions  
w_test = TestFunction(ME)   # Test function
u_test, p_test, phi_test = split(w_test)  # test fields  

#Define trial functions neede for automatic differentiation
dw = TrialFunction(ME)  

#  Define vector and scalar spaces for storing old values of velocity and acceleration
#  They are also used  later for visualization of results
W2 = FunctionSpace(mesh, U2)  # Vector space  
W  = FunctionSpace(mesh,P1)   # Scalar space  

'''''''''''''''''''''
SUBROUTINES
'''''''''''''''''''''

# Deformation gradient 
def F_calc(u):
    Id = Identity(3)            # Identity tensor
    F  = Id + grad(u)           # 3D Deformation gradient
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

#----------------------------------------------
# Subroutine for calculating the Piola  stress
#----------------------------------------------
def T_mat_calc(u, p, phi):
    Id = Identity(3) 
    F   = F_calc(u)
    J = det(F)
    #
    T_mech   = T_mech_calc(u,p)
    #
    T_maxw = T_maxw_calc(u,phi)
    #
    T      = T_mech + T_maxw
    #
    Tmat   = J * T * inv(F.T)
    return Tmat

#-----------------------------------------------------------------------------
# Define  the subroutine for calculating  the referential electric displacement
# to the Piola stress
#------------------    
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
file_results = XDMFFile("results/3D_membrane_curling.xdmf")
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
        
        T11_Maxw_Vis = project(T_maxw[0,0],W)
        T11_Maxw_Vis.rename("T11_Maxw, kPa","")
        T22_Maxw_Vis = project(T_maxw[1,1],W)
        T22_Maxw_Vis.rename("T22_Maxw, kPa","")  
        
        T11_Vis = project(T[0,0],W)
        T11_Vis.rename("T11, kPa","")
        T22_Vis = project(T[1,1],W)
        T22_Vis.rename("T22, kPa","")          
 
       # Write field quantities of interest
        file_results.write(u_Vis, t)
        file_results.write(p_Vis, t)
        file_results.write(phi_Vis, t)
        file_results.write(J_Vis, t)  
        file_results.write(Mises_Vis, t)
        file_results.write(T11_Mech_Vis, t)  
        file_results.write(T22_Mech_Vis, t)           
        file_results.write(T11_Maxw_Vis, t)  
        file_results.write(T22_Maxw_Vis, t)           
        file_results.write(T11_Vis, t)  
        file_results.write(T22_Vis, t)                   
        
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
# Boundary condition definitions
#
bcs_0 = DirichletBC(ME.sub(0).sub(0), 0, facets, 2)  # u1 fix - Left
bcs_1 = DirichletBC(ME.sub(0).sub(2), 0, facets, 2)  # u3 fix - left
#
bcs_2 = DirichletBC(ME.sub(0).sub(0), 0, facets, 3)  # u1 fix - left, bottom edge
bcs_3 = DirichletBC(ME.sub(0).sub(1), 0, facets, 3)  # u2 fix - left, bottom edge
bcs_4 = DirichletBC(ME.sub(0).sub(2), 0, facets, 3)  # u3 fix - left, bottom edge 
#
bcs_5 = DirichletBC(ME.sub(2), 0, facets, 6)  # phi ground - Bottom
bcs_6 = DirichletBC(ME.sub(2), 0, facets, 7)  # phi ground - Center
bcs_7 = DirichletBC(ME.sub(2), phiRamp, facets, 5)  # phi ramp - top
#
bcs_8 = DirichletBC(ME.sub(0).sub(2), 0, facets, 8)  # u3 fix - Front
bcs_9 = DirichletBC(ME.sub(0).sub(2), 0, facets, 9)  # u3 fix - Back

# BC set
bcs = [bcs_0, bcs_1, bcs_2, bcs_3, bcs_4, bcs_5, bcs_6, bcs_7, bcs_8, bcs_9]


# Set up the non-linear problem 
electrostaticProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
 
# Set up the non-linear solver
solver  = NonlinearVariationalSolver(electrostaticProblem)

# Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'mumps' #'mumps' # 'petsc'   #'gmres'
prm['newton_solver']['absolute_tolerance'] =  1.e-8
prm['newton_solver']['relative_tolerance'] =  1.e-8
prm['newton_solver']['maximum_iterations'] = 30

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

    
    # Store potential and displacement at this time
    timeHist0[ii] = w.sub(0).sub(1)(length, thickness/2, width/2)
    timeHist1[ii] = w.sub(2)(length/2, thickness, width/2)

    # Print progress of calculation 
    if ii%1 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: {}   |   Simulation Time: {}  s  |     Wallclock Time: {}".format(step, round(t,2), current_time))
        print("Iterations: {}".format(iter))
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

# Plot the voltage versus tip displacement
Volts = timeHist1 
#
#
plt.plot(Volts, timeHist0, c=colors[0], linewidth=1.0, marker='.')
plt.grid(linestyle="--", linewidth=0.5, color='b')
ax = plt.gca()
#
ax.set_ylabel('Displacement (mm)')
#ax.set_ylim([0.2,1.1])
#ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
#
ax.set_xlabel(r'$\phi$, kV')
#ax.set_xlim([0,1.0])
#ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
#
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
#plt.show()

fig = plt.gcf()
fig.set_size_inches(6,4)
plt.tight_layout()
plt.savefig("results/electroelastic_bilayer_actuate.png", dpi=600)
