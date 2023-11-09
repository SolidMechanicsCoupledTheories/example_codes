"""
Code for finite viscoelasticity with u-p formulation, including
implicit dynamics with the Newmark kinematic relations

Problem: Pressure-based actuation of an axisymmetric VHB "loudspeaker" membrane
         
VHB material properties taken from Wang et al., JMPS (2016).
    
Units:
Length: mm
Time  :  s
Mass  : kg
#
Mass density: kg/mm^3
Force       : mN
Stress      : kPa 
#

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

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''

radius    = 50.   # mm
thickness = 2   # mm

mesh   = RectangleMesh(Point(0.,0.),Point(radius, thickness),30,2,"crossed")

# This says "spatial coordinates" but is really the referential coordinates,
# since the mesh does not convect in FEniCS.
x = SpatialCoordinate(mesh) 

# Identify the boundary entities of the mesh
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0) and on_boundary
    
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],radius) and on_boundary
    
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], thickness) and on_boundary

# Mark boundary subdomains
facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
DomainBoundary().mark(facets, 1)  # First, mark all boundaries with common index
# Next mark specific boundaries
Left().mark(facets, 2)
Right().mark(facets,4)
Top().mark(facets,5)

# Define the boundary integration measure "ds".
ds = Measure('ds', domain=mesh, subdomain_data=facets)

# get facet normals, for applying pressure
n2D = FacetNormal(ds)

# Convert to a 3D normal vector (with z-component = 0)
n = as_vector([n2D[0], n2D[1], 0.0])

'''''''''''''''''''''''''''''''''''''''
MODEL & SIMULATION PARAMETERS
'''''''''''''''''''''''''''''''''''''''

# Material parameters
#
rho = Constant(1e-6)           # 1000 kg/m^3 = 1e-6 kg/mm^3
#
Geq_0   = Constant(15.36)      # Shear modulus, kPa
Kbulk   = 1e3*Geq_0            # Bulk modulus, kPa
lambdaL = Constant(5.99)       # Arruda-Boyce locking stretch

# Generalized-alpha method parameters for calculating acceleration and velocity
alpha   = Constant(0.2)
gamma   = Constant(0.5+alpha)
beta    = Constant(0.25*(gamma+0.5)**2)

# Viscoelasticity parameters from Wang et al (2016) for VHB4910
#
# (We have reduced Wang et al.'s branch 1 relaxation time of 0.6704 s 
#  to 5 ms in order to see dissipation on the time scales relevant 
#  to this beam oscillation problem.)
#
Gneq_1  = Constant(26.06)    #  Non-equilibrium shear modulus, kPa
tau_1   = Constant(5e-3)    #  relaxation time, s
#
Gneq_2  = Constant(26.53)    #  Non-equilibrium shear modulus, kPa
tau_2   = Constant(6.56)     #  relaxation time, s
#
Gneq_3  = Constant(10.83)    #  Non-equilinrium shear modulus, kPa
tau_3   = Constant(61.25)    #  relaxation time, s

# Simulation time control-related params
pressMax = 0.5                     # pressure amplitude, kPa
# Time history
t      = 0.0                  # start time (s)
freq   = 100                   # frequency in Hz, cycles/s
omega  = 2.*np.pi*freq        # angular frequency radians/s
nCycles= 5                    # number of cycles
Ttot   = nCycles/freq         # total time
#
preStretch = 1.05
dispMax    = (preStretch-1)*radius
preStretchTime = 1 # s
#
dispAmp = Expression(("min(dispMax*t/preStretchTime, dispMax)"), \
                     dispMax = dispMax, t = 0.0, preStretchTime = preStretchTime, degree= 1)
#
pressAmp = Expression(("t > preStretchTime ?  pressMax*sin(omega*(t - preStretchTime)) : 0 "), omega = omega,\
                      preStretchTime = preStretchTime, pressMax = pressMax, t = 0.0, degree=1)

# Time step
numSteps = 50 # per cycle
dt       = Ttot/numSteps/nCycles       # (fixed) step size
dk       = Constant(dt)

'''''''''''''''''''''
FEM SETUP
'''''''''''''''''''''
# Define function spaces
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)                 # for displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)                 # for pressure
T1 = TensorElement("Lagrange", mesh.ufl_cell(), 1, symmetry =True, shape=(3,3)) # for symmetric tensorial internal variables
#
TH = MixedElement([U2, P1])   # Mixed element. We have three internal variables
ME = FunctionSpace(mesh, TH)              # Total space for all DOFs

# Define actual functions 
w = Function(ME)
u, p  = split(w)

# A copy of  thefunctions to store values in last step for time-stepping.
w_old = Function(ME) 
u_old, p_old = split(w_old)

# Define test functions 
w_test = TestFunction(ME) 
u_test, p_test  = split(w_test) 

# Define trial functions needed for automatic differentiation
dw = TrialFunction(ME)   

# Initialization of the tensorial internal variables A1, A2, A3
# 
#  We need to first define separate functions for 
#  A1_old, A2_old, and A3_old  on the tensor function space
#  using the code below and then assign them the identity tensor:
#
T1_state = FunctionSpace(mesh, T1) 
A1_old   = Function(T1_state)
A2_old   = Function(T1_state)
A3_old   = Function(T1_state)

# Assign identity as initial value for the above functions
A1_old.assign(project(Identity(3), T1_state))
A2_old.assign(project(Identity(3), T1_state))
A3_old.assign(project(Identity(3), T1_state))

#  Define vector and scalar spaces for storing old values of velocity and acceleration
#  They are also used  later for visualization of results
W2 = FunctionSpace(mesh, U2)  # Vector space  
W  = FunctionSpace(mesh,P1)   # Scalar space  

# Functions for storing the velocity and acceleration at prev. step
v_old = Function(W2)
a_old = Function(W2)

'''''''''''''''''''''
SUBROUTINES
'''''''''''''''''''''
#---------------------------------------------------------------------
# Subroutine for updating  acceleration using the Newmark beta method
#
# The 'ufl' argument has the following uses:
#
#  ufl = True  -->  Use UFL representations of fields for the kinematic update ---
#                   representations which are symbolic & differentiable quantities. 
#                   Used in the (differentiable) inertia term of the weak form.
#
#  ufl = False --> Use simple arrays of floats for the kinematic update ---
#                   this is for updating the actual numeric value of the velocity and 
#                   acceleration at each point at the end of each step, i.e.
#                   (v_old <-- v)   (a_old <-- a).
#
# Adapted from Jeremy Bleyer's example code on elastodynamics.
#
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dk
        beta_ = beta
    else:
        dt_ = float(dk)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

#---------------------------------------------------------------------
#Subroutine for updating  velocity using the Newmark beta method
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dk
        gamma_ = gamma
    else:
        dt_ = float(dk)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

#---------------------------------------------------------------------
#  Subroutine for Updating fields at the end of each time step
def update_fields(u_proj, u_proj_old, v_old, a_old):
    
    # Get vectors (references)
    u_vec  = u_proj.vector()
    u_old_vec = u_proj_old.vector()
    v_old_vec = v_old.vector()
    a_old_vec = a_old.vector()

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u_old_vec, v_old_vec, a_old_vec, ufl=False)
    v_vec = update_v(a_vec, u_old_vec, v_old_vec, a_old_vec, ufl=False)

    # Update (v_old <- v, a_old <- a)
    v_old.vector()[:] = v_vec
    a_old.vector()[:] = a_vec


#-------------------------------------------------------
# Subroutines for kinematics and constitutive equations 
#-------------------------------------------------------
# Special gradient operators for axisymmetric functions 
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

def lambdaBar_calc(u):
    F = F_axi_calc(u)
    C = F.T*F
    Cdis = J**(-2/3)*C
    I1 = tr(Cdis)
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


# Generalized shear modulus for Arruda-Boyc model
def Geq_AB_calc(u):
    zeta = zeta_calc(u)
    Geq_AB  = Geq_0 * zeta
    return Geq_AB


#-------------------------- 
# Subroutine for calculating the  equilibrium Cauchy stress
#--------------------------    
def T_eq_calc(u,p):
    Id = Identity(3) 
    F   = F_axi_calc(u)
    J = det(F)
    B = F*F.T
    Bdis = J**(-2/3)*B
    Geq  = Geq_AB_calc(u)
    T_eq = (1/J)* Geq * dev(Bdis) - p * Id
    return T_eq

#-------------------------------------------------------------
# Subroutines for calculating non-equilibrium Cauchy stressess
#--------------------------------------------------------------
def T_neq_calc(u, A, Gneq):
    Id = Identity(3) 
    F  = F_axi_calc(u)
    J = det(F)
    C =  F.T*F
    Fdis = J**(-1/3)*F
    Cdis = J**(-2/3)*C
    fac1 = Fdis*A*Fdis.T
    fac2 = (1/3)*inner(A,Cdis)*Id
    T_neq = (1/J)*Gneq*(fac1 - fac2)
    return T_neq
    
#-------------------------------------------------------------
# Subroutines for updating internal tensor variables Ai
#--------------------------------------------------------------
def A_update(Cdis, A_old, tau):
    A_new = (1/(1+dk/tau)) * ( A_old + (dk/tau)*inv(Cdis) )
    return A_new

#----------------------------------------------
# Subroutine for calculating the Piola  stress
#----------------------------------------------
def T_mat_calc(u, p, A1, A2, A3):
    Id = Identity(3) 
    F   = F_axi_calc(u)
    J = det(F)
    T_eq   = T_eq_calc(u,p)
    T_neq1 = T_neq_calc(u, A1, Gneq_1)
    T_neq2 = T_neq_calc(u, A2, Gneq_2)
    T_neq3 = T_neq_calc(u, A3, Gneq_3)
    T      = T_eq + T_neq1 + T_neq2 + T_neq3
    Tmat   = J* T * inv(F.T)
    return Tmat

# alpha-method averaging function
def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new


'''''''''''''''''''''''''''''
Kinematics and Constitutive equations
'''''''''''''''''''''''''''''
# Get acceleration and velocity at end of step
a_new = update_a(u, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

# get avg (u,p) fields for generalized-alpha method
u_avg  = avg(u_old, u, alpha)
p_avg  = avg(p_old, p, alpha)

# Some kinematical quantities
F =  F_axi_calc(u_avg) 
J = det(F)
C =  F.T*F
Fdis = J**(-1/3)*F
Cdis = J**(-2/3)*C
lambdaBar = lambdaBar_calc(u_avg)

# Update internal tensor variables Ai
A1 = A_update(Cdis, A1_old, tau_1)
A2 = A_update(Cdis, A2_old, tau_2)
A3 = A_update(Cdis, A3_old, tau_3)

# Piola stress
Tmat = T_mat_calc(u_avg, p_avg, A1, A2, A3)

'''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of linear momentum  
# Res_1: Pressure variable  

# The weak form for the equation of motion
#
# Configuration-dependent traction
Fcof       = J*inv(F.T) #  Cofactor of F
traction   = - pressAmp*dot(Fcof,n) # configuration-dependent pressure traction vector
#
Res_0 =  inner( Tmat, axi_grad_vector(u_test) )*x[0]*dx \
              - dot(traction, as_vector([u_test[0], u_test[1], 0]))*x[0]*ds(5) \
               + inner(rho * a_new, u_test)*x[0]*dx 

# The weak form for the pressure  
Res_1 =  dot((p_avg/Kbulk + ln(J)/J) , p_test)*x[0]*dx

# total weak form
Res  = Res_0 + Res_1

# Automatic differentiation tangent:
a = derivative(Res, w, dw)
   

'''''''''''''''''''''
 SET UP OUTPUT FILES
'''''''''''''''''''''

# Output file setup
file_results = XDMFFile("results/viscoelastic_speaker_membrane.xdmf")
# "Flush_output" permits reading the output during simulation
# (Although this causes a minor performance hit)
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Subroutine for writing results  to XDMF at time t
def writeResults(t):
  
    # Variable projecting and renaming
    u_Vis = project(u, W2)
    u_Vis.rename("disp"," ")
     
    # Visualize the pressure
    p_Vis = project(p, W)
    p_Vis.rename("p"," ")
    
    # Project internal varaible A1  for visualization
    A1_11_Vis = project(A1[0,0],W)
    A1_11_Vis.rename("A1_11"," ")
    #
    A1_22_Vis = project(A1[1,1],W)
    A1_22_Vis.rename("A1_22"," ")
    #
    A1_33_Vis = project(A1[2,2],W)
    A1_33_Vis.rename("A1_33"," ")
    
    # Project  Piola stress for visualization
    TR11_Vis = project(Tmat[0,0],W)
    TR11_Vis.rename("TR11"," ")
    
    TR22_Vis = project(Tmat[1,1],W)
    TR22_Vis.rename("TR22"," ")  
    
    TR33_Vis = project(Tmat[2,2],W)
    TR33_Vis.rename("TR33"," ")  
    
    # Visualize effective stretch
    lambdaBar_Vis = project(lambdaBar,W)
    lambdaBar_Vis.rename("LambdaBar"," ")
    
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
    file_results.write(TR11_Vis, t)
    file_results.write(TR22_Vis, t) 
    file_results.write(TR33_Vis, t) 
    file_results.write(A1_11_Vis, t)
    file_results.write(A1_22_Vis, t) 
    file_results.write(A1_33_Vis, t) 
    file_results.write(lambdaBar_Vis, t)
    file_results.write(Mises_Vis, t)
        
# Write initial state to XDMF file
writeResults(t=0)

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
BOUNDARY CONDITIONS
'''''''''''''''''''''''

# Boundary condition definitions
bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 2)  # u1 fix - Left
#
bcs_2 = DirichletBC(ME.sub(0).sub(0), dispAmp, facets, 4)  # u1 disp - Right
bcs_3 = DirichletBC(ME.sub(0).sub(1), 0., facets, 4)  # u2 fix - Right

# BC set
bcs = [bcs_1, bcs_2, bcs_3]


#########################

# Set up the non-linear problem 
stressProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
 
# Set up the non-linear solver
solver  = NonlinearVariationalSolver(stressProblem)

#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance']   = 1.e-8
prm['newton_solver']['relative_tolerance']   = 1.e-7
prm['newton_solver']['maximum_iterations']   = 25
prm['newton_solver']['relaxation_parameter'] = 1.0
# prm['newton_solver']['error_on_nonconvergence'] = False

# Initialize time history variables.
totSteps = 100000
timeHist0 = np.zeros(shape=[totSteps])
timeHist1 = np.zeros(shape=[totSteps]) 
timeHist2 = np.zeros(shape=[totSteps]) 

# initialize the step counter
ii = 0 # counter 

# Time-stepping solution procedure 
while (round(t, 9) <= (Ttot + preStretchTime)):
    
    # Take larger steps during the pre-stretching
    if t + dt > preStretchTime:
        dt = Ttot/numSteps/nCycles       # (fixed) step size
        dk.assign(dt)
    else:
        dt = 0.1
        dk.assign(dt)
            
    # increment time
    t += dt

    # update time variables in time-dependent BCs
    dispAmp.t  = t - float(alpha)*dt
    pressAmp.t = t - float(alpha)*dt
    
    # increment counter
    ii  += 1

    # Solve the problem
    (iter, converged) = solver.solve()
    
    # Write results to XDMF file
    writeResults(t)
    
    # Calculate the pressure force at the moved boundary:  
    traction = dot(Tmat,n)
    Force    = dot(traction,n)*ds(5)
    
    # Store force and displacement at this time
    timeHist0[ii] = w.sub(0).sub(1)(0.0,0.0) # midspan displacement, mm
    # timeHist1[ii] = assemble(Force) #pressMax*sin(omega*t) # pressure applied, kPa
    timeHist1[ii] = pressMax*sin(omega*(t- preStretchTime)) # pressure applied, kPa
    timeHist2[ii] = t - preStretchTime # current time
    
    # Update fields for next step
    #
    # First, we must update the velocity and acceleration
    # ( v -> v_old, a -> a_old )
    u_proj = project(u, W2)
    u_proj_old = project(u_old, W2)
    update_fields(u_proj, u_proj_old, v_old, a_old)
    #
    # Now the degrees of freedom:
    w_old.vector()[:] = w.vector()
    # 
    # Now the tensorial state variables:
    A1_old.assign(project(A1, T1_state))
    A2_old.assign(project(A2, T1_state))
    A3_old.assign(project(A3, T1_state))

   # Print progress of calculation 
    if ii%1 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: {}  |  Simulation Time: {}  s  |  Wallclock Time: {}".format(step, round(t,3), current_time))
        print("Iterations: {}".format(iter))
        print()

# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("--------------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("--------------------------------------------")


'''''''''''''''''''''
    VISUALIZATION
'''''''''''''''''''''
# Set plot font to size 14
font = {'size'   : 14}
plt.rc('font', **font)

# Get array of default plot colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors     = prop_cycle.by_key()['color']

# Only plot as far as we have time history data
ind = np.argmax(timeHist2) + 1

fig, (ax1, ax2) = plt.subplots(2,1, sharex='col')

ax2.plot(timeHist2[0:ind], timeHist1[0:ind], c=colors[3], linewidth=1.0, marker='.')
ax2.grid(linestyle="--", linewidth=0.5, color='b')
ax2.set_ylabel('Pressure (kPa)')
ax2.set_xlabel('Time (s)')
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.set_xlim(0, Ttot)

ax1.plot(timeHist2[0:ind], -timeHist0[0:ind], c=colors[0], linewidth=1.0, marker='.')
ax1.grid(linestyle="--", linewidth=0.5, color='b')
ax1.set_ylabel('Displacement (mm)')
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
#plt.show()

fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/viscoelastic_speaker_membrane.png", dpi=600)


"""
# MOVIE GENERATION:
from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams['animation.ffmpeg_path'] = 'media/ffmpeg'

plt.figure()
fig, (ax1, ax2) = plt.subplots(2,1, sharex='col')
fig.set_size_inches(7,6)
    
def animate(i):
    plt.cla()
    
    ax1.plot(timeHist2[0:i], timeHist1[0:i], c=colors[3], linewidth=1.5)
    ax1.grid(linestyle="--", linewidth=0.5, color='b')
    ax1.set_ylabel('Pressure (kPa)')
    from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_ylim(-0.75,0.75)
    ax1.set_xlim(0, 5)
    
    ax2.plot(timeHist2[0:i], timeHist0[0:i], c=colors[0], linewidth=1.5)
    ax2.grid(linestyle="--", linewidth=0.5, color='b')
    ax2.set_ylabel('Displacement (mm)')
    ax2.set_xlabel('Time (s)')
    from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_ylim(-12.5, 12.5)

set_fps = 24
ani = FuncAnimation(fig, animate, interval=1000/set_fps, save_count=ind+1)

f = r"media/viscoelastic_speaker_membrane_curve.mp4"
writervideo = FFMpegWriter(fps=set_fps) 
ani.save(f, writer=writervideo)#, dpi=200)
"""




