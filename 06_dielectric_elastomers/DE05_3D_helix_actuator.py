"""
3D Code for finite electro-elasticity.

Problem: electro-elastic actuation of a helical actuator
    
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


'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''
# Initialize an empty mesh object
mesh = Mesh()

# Read the *.xdmf file data into mesh object
with XDMFFile("meshes/helix.xdmf") as infile:
    infile.read(mesh)

# Read the  subdomain data stored in the facet_*.xdmf file
mvc2d = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("meshes/facet_helix.xdmf") as infile:
    infile.read(mvc2d, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc2d)


# Note the surface numbering, from the gmsh *.geo file:
# // Base set    - 539
# // Outside set - 540
# // Inside set  - 541
# // Volume set  - 542

# Identify a point to be  possibly fixed
def Ground(x, on_boundary):
        return near(x[0],5.1) and near(x[1], 0) and near(x[2], 0)

# This says "spatial coordinates" but is really the referential coordinates,
# since the mesh does not convect in FEniCS.
x = SpatialCoordinate(mesh) 

# Define surface area measure
ds = Measure("ds", domain=mesh, subdomain_data=mf)

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
# Mechanical parameters
Geq_0   = 15         # Shear modulus, kPa
Kbulk   = 1e3*Geq_0  # Bulk modulus, kPa
I_m     = 175         # Gent locking paramter
# Electrostatic  parameters
vareps_0 = Constant(8.85E-3)         #  permittivity of free space pF/mm
vareps_r = Constant(5)             #  relative permittivity, dimensionless
vareps   = vareps_r*vareps_0         #  permittivity of the material

# Simulation time control-related params
t        = 0.0         # start time (s)
phiTot   = 2.0         # final phi (number in kV)
Ttot     = 1           # Total time (s)
dt       = 0.01        # (fixed) step size
dk       = Constant(dt)

# Boundary condition to ramp up electrostatic potential
phiRamp = Expression(("phiTot*t/tRamp"),
                    t = 0.0, phiTot = phiTot, tRamp=Ttot, degree=1)

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

 
# Piola  stress
def T_mat_calc(u, p, phi):
    Id = Identity(3) 
    F   = F_calc(u)
    J = det(F)
    #
    T_mech  = T_mech_calc(u,p)
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
file_results = XDMFFile("results/helix_actuate_n1.xdmf")
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

# Note the surface numbering, from the gmsh *.geo file:
# // Base set    - 539
# // Outside set - 540
# // Inside set  - 541
# // Volume set  - 542

# Boundary condition definitions
#
bcs_0 = DirichletBC(ME.sub(0), Constant((0,0,0)), mf, 539) # u built in -- base
#
bcs_1 = DirichletBC(ME.sub(2), 0, mf, 541)         # phi ground - inside
bcs_2 = DirichletBC(ME.sub(2), phiRamp, mf, 540)  # phi ramp - outside
#


# BC set
bcs = [bcs_0, bcs_1, bcs_2]
'''''''''''''''''''''''
Define  and solve the nonlinear variational problem
'''''''''''''''''''''''
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
prm['newton_solver']['error_on_nonconvergence'] = False

# Initalize output array for tip displacement
timeHist0 = np.zeros(shape=[100000])
timeHist1 = np.zeros(shape=[100000]) 
timeHist2 = np.zeros(shape=[100000]) 
#Iinitialize a counter for reporting data
ii=0

# Time-stepping solution procedure loop
while (round(t + dt, 9) <= Ttot):
    
    # increment time
    t += float(dt)
    
    # update time variables in time-dependent BCs 
    phiRamp.t = t 

    # Solve the problem
    (iter, converged) = solver.solve()  
         
    # Now we start the adaptive time-stepping and output storage procedure.
    #
    # First, we check if the newton solver actually converged.
    if converged: 
        
        # If the solver converged, we print the status of the solver, 
        # perform adaptive time-stepping updates, output results, and 
        # update degrees of freedom for the next step, w_old <- w.
        
        # increment counter
        ii += 1
        
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
        if iter<=3:
            dt = 1.5*dt
            dk.assign(dt)
        # If the newton solver takes 5 or more iterations, 
        # decrease the time step by a factor of 2:
        elif iter>=9:
            dt = dt/2
            dk.assign(dt)
        # otherwise the newton solver took 3-4 iterations,
        # in which case leave the time step alone.
            
        # Write output to *.xdmf file
        writeResults(t)
        
        # Store  displacement and potential at a particular point  at this time
        IntTol = 1e-3 # tolerance for interpolating surface quantities, mesh is not totally precise.
        timeHist0[ii] = w.sub(0).sub(1)(5.1-IntTol, -IntTol, 35-IntTol) # time history of y-displacement
                                                                        # top outside point of the helix
        timeHist1[ii] = w.sub(2)(5.1-IntTol, -IntTol, 35-IntTol)  # time history of voltage (phi)
        timeHist2[ii] = t # current time
        
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

# Only plot as far as we have time history data
ind = np.argmax(timeHist2) + 1

fig, (ax1, ax2) = plt.subplots(2,1, sharex='col')

ax1.plot(timeHist2[0:ind], timeHist1[0:ind], c=colors[3], linewidth=1.0, marker='.')
ax1.grid(linestyle="--", linewidth=0.5, color='b')
ax1.set_ylabel('Electric potential (kV)')
# ax1.set_xlim([0,5])
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())

ax2.plot(timeHist2[0:ind], timeHist0[0:ind], c=colors[0], linewidth=1.0, marker='.')
ax2.grid(linestyle="--", linewidth=0.5, color='b')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Displacement (mm)')
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
#plt.show()

fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/helix_actuate_n1.png", dpi=600)
