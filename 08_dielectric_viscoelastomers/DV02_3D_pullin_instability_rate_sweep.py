"""
Code for finite electro-viscoelasticity of dielectric elastomers 
    with u-p formulation, including implicit dynamics with 
    the Newmark kinematic relations

Problem: Electro-viscoelastic pull-in instability of a VHB unit cube
            at various rates, to show rate-sensitivity of the 
            instability onset.
         
VHB material properties taken from Wang et al., JMPS (2016).
    
Units:
Length: mm
Time  :  s
Mass  : kg
Charge: nC
#
Mass density       : kg/mm^3
Force              : mN
Stress             : kPa 
Electric Potential : kV 
#

Eric M. Stewart    and    Lallit Anand   
(ericstew@mit.edu)        (anand@mit.edu)   

 November 2023
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
# Create mesh and identify the 3D domain and its boundary
#
# Box size, in mm
L= 1.0 
# A basic box mesh 
mesh = BoxMesh(Point(0.,0.,0.),Point(L,L,L),2,2,2)

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
 
# Define the boundary integration measure "ds"
ds = Measure("ds", domain=mesh, subdomain_data=facets)

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
alpha   = Constant(0.0)
gamma   = Constant(0.5+alpha)
beta    = Constant(0.25*(gamma+0.5)**2)

# Viscoelasticity parameters from Wang et al (2016) for VHB4910
#
Gneq_1  = Constant(26.06)    #  Non-equilibrium shear modulus, kPa
tau_1   = Constant(0.6074)    #  relaxation time, s
#
Gneq_2  = Constant(26.53)    #  Non-equilibrium shear modulus, kPa
tau_2   = Constant(6.56)     #  relaxation time, s
#
Gneq_3  = Constant(10.83)    #  Non-equilinrium shear modulus, kPa
tau_3   = Constant(61.25)    #  relaxation time, s

# Electrostatic  parameters
vareps_0 = Constant(8.85E-3)         #  permittivity of free space pF/mm
vareps_r = Constant(4.8)             #  relative permittivity, dimensionless
vareps   = vareps_r*vareps_0         #  permittivity of the material

# for rampRate in [1e-5, 1e-3, 1e-2, 1e-1, 1e0, 1e1]:
    
for rampRate in [1e-5, 1e1]:
        
    # Simulation time control-related params
    t        = 0.0           # start time (s)
    Ttot     = 1.0/rampRate  # total simulation time (s) 
    numSteps = 250
    dt       = Ttot/numSteps       # (fixed) step size
    dk       = Constant(dt)
    
    # Normalization parameter for voltage is l*sqrt(Geq_0/vareps)
    #
    phiTot = 1.6* float(L*np.sqrt(float(Geq_0)/float(vareps)))  # final normalized value of phi
    
    # Boundary condition to ramp up electrostatic potential
    phiRamp = Expression(("phi_tot*t/Ttot"), 
                          t = 0.0, phi_tot = phiTot, Ttot=Ttot, degree=1)
    
    
    '''''''''''''''''''''
    FEM SETUP
    '''''''''''''''''''''
    # Define function spaces
    U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)                 # for displacement
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)                 # for pressure
    T1 = TensorElement("Lagrange", mesh.ufl_cell(), 1, symmetry =True) # for symmetric tensorial internal variables
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
    # Deformation gradient 
    def F_calc(u):
        Id = Identity(3) 
        F = Id + grad(u) 
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
        F   = F_calc(u)
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
        F  = F_calc(u)
        J = det(F)
        C =  F.T*F
        Fdis = J**(-1/3)*F
        Cdis = J**(-2/3)*C
        fac1 = Fdis*A*Fdis.T
        fac2 = (1/3)*inner(A,Cdis)*Id
        T_neq = (1/J)*Gneq*(fac1 - fac2)
        return T_neq
        
        
    # Subroutine for calculating the electrotatic contribution to the Cauchy stress
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
    def T_mat_calc(u, p, A1, A2, A3, phi):
        Id = Identity(3) 
        F   = F_calc(u)
        J = det(F)
        #
        T_eq   = T_eq_calc(u,p)
        #
        T_neq1 = T_neq_calc(u, A1, Gneq_1)
        T_neq2 = T_neq_calc(u, A2, Gneq_2)
        T_neq3 = T_neq_calc(u, A3, Gneq_3)
        #
        T_maxw = T_maxw_calc(u,phi)
        #
        T = T_eq + T_neq1 + T_neq2 + T_neq3 + T_maxw
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
        
        
    #-------------------------------------------------------------
    # Subroutines for updating internal tensor variables Ai
    #--------------------------------------------------------------
    def A_update(Cdis, A_old, tau):
        A_new = (1/(1+dk/tau)) * ( A_old + (dk/tau)*inv(Cdis) )
        return A_new
    
    
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
    phi_avg  = avg(phi_old, phi, alpha)
    
    # Some kinematical quantities
    F =  F_calc(u_avg) 
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
    Tmat = T_mat_calc(u_avg, p_avg, A1, A2, A3, phi_avg)
    
    # Referential electric displacement
    Dmat = Dmat_calc(u_avg,phi_avg)
    
    '''''''''''''''''''''''
    WEAK FORMS
    '''''''''''''''''''''''
    # Residuals:
    # Res_0: Balance of linear momentum  
    # Res_1: Pressure variable  
    # Res_2: Gauss's law  
    
    # The weak form for the equation of motion
    #
    Res_0 =  inner( Tmat, grad(u_test) )*dx \
              + inner(rho * a_new, u_test)*dx 
    
    # The weak form for the pressure  
    Res_1 =  dot((p_avg/Kbulk + ln(J)/J) , p_test)*dx
    
    #  The weak form for Gauss's Law
    Res_2 = inner(Dmat, grad(phi_test))*dx 
    
    # Total weak form
    Res  =  Res_0 + Res_1 + Res_2
    
    # Automatic differentiation tangent:
    a = derivative(Res, w, dw)
       
    
    '''''''''''''''''''''
     SET UP OUTPUT FILES
    '''''''''''''''''''''
    
    # Output file setup
    file_results = XDMFFile("results/electro_viscoelastic_3D_pullin.xdmf")
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
    
        # Visualize  normalized chemical potential
        phi_Vis = project(phi,W)
        phi_Vis.rename("phi, kV"," ")   
        
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
        file_results.write(phi_Vis, t)
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
    
    # Recall the boundary definitions:
    #
    # xBot().mark(facets, 1)
    # xTop().mark(facets, 2)
    # yBot().mark(facets, 3)
    # yTop().mark(facets, 4)
    # zBot().mark(facets, 5)
    # zTop().mark(facets, 6)
    
    # Boundary condition definitions, homogeneous 
    bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 1)  # u1 fixed - xBot
    bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 3)  # u2 fixed - yBot
    bcs_3 = DirichletBC(ME.sub(0).sub(2), 0, facets, 5)  # u3 fixed - zBot
    #
    bcs_4 = DirichletBC(ME.sub(2), 0, facets, 3)           #  phi ground bottom, face 3
    bcs_5 = DirichletBC(ME.sub(2), phiRamp, facets, 4)     #  phi ramp top, face 6 
    
    # BC set
    bcs = [bcs_1, bcs_2, bcs_3, bcs_4, bcs_5]
    
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
    prm['newton_solver']['error_on_nonconvergence'] = True
    
    # Initialize time history variables.
    totSteps = 100000
    timeHist0 = np.zeros(shape=[totSteps])
    timeHist1 = np.zeros(shape=[totSteps]) 
    timeHist2 = np.zeros(shape=[totSteps]) 
    
    # initialize the step counter
    ii = 0 # counter 
    
    # Time-stepping solution procedure 
    while (round(t + dt, 9) <= Ttot):
        
        # increment time
        t += dt
    
        # update time variables in time-dependent BCs 
        phiRamp.t = t - float(alpha)*dt
        
        # increment counter
        ii  += 1
    
        # Solve the problem
        try:
            (iter, converged) = solver.solve()
        except: # Break the loop if solver fails
            break
        
        # Write results to XDMF file
        writeResults(t)
        
        # Store the displacement and force at this time
        timeHist0[ii] = w.sub(0).sub(1)(L, L, L) # time history of displacement
        timeHist1[ii] = w.sub(2)(L, L, L)        # time history of voltage phi
        timeHist2[ii] =  t # current time
        
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
            print("Step: {}  |  Simulation Time: {}  s  |  Wallclock Time: {}".format(step, round(t,2), current_time))
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
    
    # set plot font to size 14
    font = {'size'   : 14}
    plt.rc('font', **font)
    
    # Get array of default plot colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # Plot the normalized dimensionless quantity for $\phi$ used in Wang et al. 2016
    # versus stretch in the vertical direction.
    #
    normVolts = (timeHist1)/L*(np.sqrt(float(vareps)/float(Geq_0)))
    normVolts = normVolts[0:ii]
    #
    stretch = timeHist0/L + 1.0
    stretch = stretch[0:ii]
    #
    if rampRate < 1e-4:
        plt.plot(normVolts, stretch, linewidth=1.0, marker='.', label=r"Equilibrium" )
    else:
        plt.plot(normVolts, stretch, linewidth=1.0, marker='.', label=r"{:1.0e}".format(rampRate) )
    plt.scatter(normVolts[ii-1], stretch[ii-1], c='k', marker='x', s=100)
    plt.grid(linestyle="--", linewidth=0.5, color='b')
    ax = plt.gca()
    #
    ax.set_ylabel(r'$\lambda$')
    ax.set_ylim([0.2,1.1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    #
    ax.set_xlabel(r'$(\phi/\ell_0)  /  \sqrt{G_{0,eq} / \varepsilon} $')
    ax.set_xlim([0,1.6])
    ax.set_xticks([0.0, 0.5, 1.0, 1.5])
    #
    from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #plt.show()

# plt.legend()
fig = plt.gcf()
fig.set_size_inches(6,5)
plt.tight_layout()
plt.savefig("results/electro-viscoelastic_pullin_instability_rate_sweep.png", dpi=600)


