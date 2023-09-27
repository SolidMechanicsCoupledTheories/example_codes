"""
Code for plane-strain  coupled thermoelasticity of elastomers

Bending of a thermoelastic bilayer
The bottom layer expands on heating
The top layer does not expand on heating

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

# A 3-D cube
length    = 100.0 # mm
thickness = 1 #mm
mesh = BoxMesh(Point(0.,0.,0.),Point(length,length,thickness),10,10,2)

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
        return near(x[2],thickness) and on_boundary
    

# Identify the corners of the domain which will be fixed  
def pin1(x, on_boundary):
        return near(x[0],0)  and near(x[1], 0) 
    
def pin2(x, on_boundary):
        return near(x[0],length)  and near(x[1], length) 
    
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
n = FacetNormal(ds)

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
# Constants for both materials
k_B      = Constant(1.38E-17)              # Boltzmann's constant
theta0   = Constant(273)                   # Initial temperature
Gshear_0 = Constant(280)                   # Ground sate shear modulus
N_R      = Constant(Gshear_0/(k_B*theta0)) # Number polymer chains per unit ref. volume             # Number polymer chains per unit ref. volume
lambdaL  = Constant(5.12)                  # Locking stretch
Kbulk    = Constant(1000.0*Gshear_0)       # Bulk modulus
c_v      = Constant(1930)                  # Specific heat
k_therm  = Constant(0.16E3)                # Thermal conductivity
alpha    = Constant(180.0E-6)              # Coefficient of thermal expansion

"""
Simulation time control-related parameters
"""
t    = 0.0   # initialization of time
Ttot = 100   # total simulation time 
dt   = 1     # Fixed step size

# change in temperature
for delta_theta in [-50, 0, 50]:
    t= 0
    #delta_theta = -50 
    
    # Boundary condition expressions as necessary
    thetaRamp = Expression(("theta0 + deltaTheta*t/tRamp"),
                    theta0 = theta0, deltaTheta = delta_theta, tRamp = Ttot, t = 0.0, degree=1)
    
    # Maximum  internal pressure
    press_max = 9.2e-3 # kPa
    
    # Expression for the time-dependent traction BC
    pressRamp = Expression("press_max*t/tRamp", \
                          press_max=press_max, t= 0.0, tRamp = Ttot, degree=1)
    
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
    
    # Deformation gradient 
    def F_calc(u):
        Id = Identity(3) 
        F  = Id + grad(u) 
        return F
    
    def lambdaBar_calc(u):
        F    = F_calc(u)
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
        z    = conditional(gt(z,0.95), 0.95, z) # Prevent the function from blowing up
        beta = z*(3.0 - z**2.0)/(1.0 - z**2.0)
        zeta = (lambdaL/(3*lambdaBar))*beta
        return zeta
    
    
    # Piola stress 
    def Tmat_calc(u, p, theta):
        F = F_calc(u)
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
        F   = F_calc(u) 
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
        F = F_calc(u) 
        J = det(F)         
        #
        Cinv = inv(F.T*F) 
        #
        Tcond = J * k_therm * Cinv # Thermal conductivity tensor
        #
        Qmat = - Tcond * grad(theta)
        return Qmat
    
    '''''''''''''''''''''''''''''
    Evaluate kinematics and constitutive relations
    '''''''''''''''''''''''''''''
    
    # Kinematics
    F = F_calc(u)
    J = det(F)   
    #
    lambdaBar = lambdaBar_calc(u)
    #
    F_old = F_calc(u_old)
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
    
    # Time step field, constant within body
    dk = Constant(dt)
    
    # Cofactor of F
    Fcof = J*inv(F.T)
    
    # Configuration-dependent traction
    traction = - pressRamp*dot(Fcof,n)
    
    # The weak form for the equilibrium equation
    Res_0 = inner(Tmat , grad(u_test) )*dx  - dot(traction, u_test)*ds(6) 
    
    # The weak form for the pressure
    fac_p =  ( ln(J) - 3*alpha*(theta-theta0) )/J
    #
    Res_1 = dot( (p/Kbulk + fac_p), p_test)*dx
          
    
    # The weak form for heat equation
    Res_2 = dot( c_v*(theta - theta_old), theta_test)*dx \
            -  (1/2)*theta * inner(M, (C - C_old)) * theta_test*dx \
            -  dk* dot(Qmat , grad(theta_test) )*dx   
            
    # Total weak form
    Res = Res_0 +  Res_1 + Res_2 
    
    # Automatic differentiation tangent:
    a = derivative(Res, w, dw)
    
    '''''''''''''''''''''''
    DIRICHLET BOUNDARY CONDITIONS
    '''''''''''''''''''''''      
    # # Next mark specific boundaries
    # xBot().mark(facets, 1)
    # xTop().mark(facets, 2)
    
    # yBot().mark(facets, 3)
    # yTop().mark(facets, 4)
    
    # zBot().mark(facets, 5)
    # zTop().mark(facets, 6)
    
    
    # disp_max = 1
     
    # # Expression for the time-dependent traction BC
    # dispRamp = Expression(("disp_max*t/tRamp", "0", "0"), \
    #                       disp_max=disp_max, t= 0.0, tRamp = Ttot, degree=1)
        
        
    # Fix the bottom left corner
    bcs_0 = DirichletBC(ME.sub(0),  Constant((0, 0, 0)), pin1, method='pointwise')
    bcs_1 = DirichletBC(ME.sub(0),  Constant((0, 0, 0)), pin2, method='pointwise')
    #
    bcs_2 = DirichletBC(ME.sub(0).sub(0), Constant(0.), facets, 1)  # u1 fix - xBot 
    bcs_3 = DirichletBC(ME.sub(0).sub(1), Constant(0.), facets, 3)  # u2 fix - yBot 
    #
    bcs_4 = DirichletBC(ME.sub(2), thetaRamp, facets, 5)  # temperature ramp - zBot
    bcs_5 = DirichletBC(ME.sub(2), thetaRamp, facets, 6)   # temperature ramp - zTop
    # 
    bcs = [bcs_0, bcs_1, bcs_2, bcs_3, bcs_4, bcs_5]
    
    # # Fix the bottom left corner
    # bcs_0 = DirichletBC(ME.sub(0), Constant((0, 0, 0)), facets, 1)  # u1 fix - xBot 
    # bcs_1 = DirichletBC(ME.sub(0), Constant((0, 0, 0)), facets, 2)  # u1 fix - xBot 
    # bcs_2 = DirichletBC(ME.sub(0), Constant((0, 0, 0)), facets, 3)  # u1 fix - xBot 
    # bcs_3 = DirichletBC(ME.sub(0), Constant((0, 0, 0)), facets, 4)  # u1 fix - xBot 
    # #
    # bcs_4 = DirichletBC(ME.sub(2), thetaRamp, facets, 5)  # temperature ramp - zBot
    # bcs_5 = DirichletBC(ME.sub(2), thetaRamp, facets, 6)   # temperature ramp - zTop
    # # 
    # bcs = [bcs_0, bcs_1, bcs_2, bcs_3, bcs_4, bcs_5]
    
    """
    SETUP NONLINEAR PROBLEM
    """
    ThermoelasProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
    solver  = NonlinearVariationalSolver(ThermoelasProblem)
    
    #Solver parameters
    prm = solver.parameters
    prm['nonlinear_solver'] = 'newton'
    prm['newton_solver']['linear_solver'] = "mumps" 
    prm['newton_solver']['absolute_tolerance'] = 1.e-6
    prm['newton_solver']['relative_tolerance'] = 1.e-6
    prm['newton_solver']['maximum_iterations'] = 30
    
    '''''''''''''''''''''
     SET UP OUTPUT FILES
    '''''''''''''''''''''
    # Output file setup
    file_results = XDMFFile("results/3D_thermoelas_solar_sail.xdmf")
    # "Flush_output" permits reading the output during simulation
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
         STEP
    """""""""""""""""
    # Give the step a descriptive name
    step = "Press/temp loading"
      
    # set up output arrays for history variables
    siz  = 100000 
    temp_out = np.zeros(siz)
    disp_out = np.zeros(siz)
    time_out = np.zeros(siz)
    temp_out[0] = theta0# initial temp
        
    # Initialize conter for reporting data
    ii = 0
    
    while (t < Ttot):
        # increment time
        t += dt
        
        # increment counter
        ii += 1
        
        # update time variables in time-dependent BCs
        thetaRamp.t = t
        pressRamp.t = t
        # dispRamp.t = t
    
        # Solve the problem
        (iter, converged) = solver.solve()  
    
        # Write output to *.xdmf file
        writeResults(t)
        
        # Update DOFs for next step
        w_old.vector()[:] = w.vector()
         
        # Write time history variables
        temp_out[ii] = w.sub(2)(length,length, 0)    # output surface temperature
        disp_out[ii] = w.sub(0).sub(2)(length, 0, 0) # output tip displacement
        time_out[ii] = t # current time
        
        # Print progress of calculation
        if ii%1 == 0:      
           now = datetime.now()
           current_time = now.strftime("%H:%M:%S")
           print("Step: {} | Simulation Time: {} s, Wallclock Time: {}".\
                 format(step, round(t,4), current_time))
           print("Iterations: {}".format(iter))
           print()       
            
    # End analysis
    print("------------------------------------")
    print("End computation")                 
    # Report elapsed real time for whole analysis
    endTime = datetime.now()
    elapseTime = endTime - startTime
    print("------------------------------------")
    print("Elapsed real time:  {}".format(elapseTime))
    print("------------------------------------")
    
    
    '''''''''''''''''''''
        VISUALIZATION
    '''''''''''''''''''''
    
    # Set up font size, initialize colors array
    font = {'size'   : 14}
    plt.rc('font', **font)
    #
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors     = prop_cycle.by_key()['color']
    
    # Only plot as far as we have time history data
    ind = np.argmax(time_out)
    
    # Create figure for temperature-displacement curve.
    #
    # fig = plt.figure() 
    plt.plot( press_max*time_out[0:ind]/Ttot*1000, disp_out[0:ind], linewidth=1.0, marker='.',\
             label=r"$\Delta\,\vartheta$ = {:1.0f}".format(delta_theta) )
    #-------------------------------------------------------------------------------
    plt.grid(linestyle="--", linewidth=0.5, color='b')
    fig = plt.gcf()
    ax = fig.gca()  
    ax.set_xlabel("Pressure, Pa",size=14)
    ax.set_ylabel("Max membrane displacement",size=14)
    ax.set_title("Membrane deflection curve", size=14, weight='normal')
    #
    from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #plt.legend()
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    
plt.legend()
# plt.show()
# Save figure to file
fig = plt.gcf()
fig.set_size_inches(7,5)
plt.tight_layout()
plt.savefig("results/3D_solar_sail_plotting.png", dpi=600) 


