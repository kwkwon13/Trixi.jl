using OrdinaryDiffEqLowStorageRK
using Trixi


dg = DGMulti(polydeg = 1,
             element_type = Quad(),
             approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())


advection_velocity = (1.5, 1.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
diffusivity() = 5.0e-2
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)


function initial_condition_diffusive_convergence_test(x, t, equation)
    RealT = eltype(x)
    x_trans = x - equation.advection_velocity * t

    nu = diffusivity()
    c = 1
    A = 0.5f0
    L = 2
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f
    scalar = c + A * sin(omega * sum(x_trans)) * exp(-2 * nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test


cells_per_dimension = (16, 16)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (-1.0, -1.0),
                   coordinates_max = (1.0, 1.0),
                   periodicity = true)


penalty_parameter = 10.0
solver_parabolic = ViscousFormulationSymmetricInteriorPenalty(penalty_parameter)



semi = SemidiscretizationHyperbolicParabolic(
    mesh,
    (equations, equations_parabolic),
    initial_condition,
    dg;
    solver_parabolic = solver_parabolic
)

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback, save_solution)

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
