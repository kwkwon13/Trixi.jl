using OrdinaryDiffEqLowStorageRK
using Trixi

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 5.0e-2
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

dg = DGMulti(polydeg = 3,
             element_type = Line(),
             approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())

cells_per_dimension = (16,)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (-convert(Float64, pi),),
                   coordinates_max = (convert(Float64, pi),),
                   periodicity = true)

function x_trans_periodic(x, domain_length = SVector(oftype(x[1], 2 * pi)),
                          center = SVector(oftype(x[1], 0)))
    x_normalized = x .- center
    x_shifted = x_normalized .% domain_length
    x_offset = ((x_shifted .< -0.5f0 * domain_length) -
                (x_shifted .> 0.5f0 * domain_length)) .*
               domain_length
    return center + x_shifted + x_offset
end

function initial_condition_diffusive_convergence_test(x, t,
                                                      equation::LinearScalarAdvectionEquation1D)
    x_trans = x_trans_periodic(x - equation.advection_velocity * t)

    nu = diffusivity()
    c = 1
    A = 0.5f0
    omega = 1
    scalar = c + A * sin(omega * sum(x_trans)) * exp(-nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

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
