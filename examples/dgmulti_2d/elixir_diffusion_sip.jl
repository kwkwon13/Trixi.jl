using OrdinaryDiffEqLowStorageRK
using Trixi

# Pure diffusion problem to demonstrate SIP method
# This example uses a lower-order polynomial and smaller penalty parameter
# for numerical stability

dg = DGMulti(polydeg = 2, element_type = Quad(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_central),
             volume_integral = VolumeIntegralWeakForm())

# Pure diffusion (zero advection velocities)
equations = LinearScalarAdvectionEquation2D(0.0, 0.0)
equations_parabolic = LaplaceDiffusion2D(1.0, equations)

# Initial condition: smooth Gaussian
function initial_condition_gaussian(x, t, equations::LinearScalarAdvectionEquation2D)
    return SVector(exp(-10 * (x[1]^2 + x[2]^2)))
end
initial_condition = initial_condition_gaussian

cells_per_dimension = (8, 8)
mesh = DGMultiMesh(dg, cells_per_dimension, periodicity = true)

# Use Symmetric Interior Penalty (SIP) method with moderate penalty parameter
# For pure diffusion with smooth initial conditions, alpha = 2.0 provides
# a good balance between stability and accuracy
solver_parabolic = ViscousFormulationSIP(2.0)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, dg;
                                             solver_parabolic = solver_parabolic)

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
