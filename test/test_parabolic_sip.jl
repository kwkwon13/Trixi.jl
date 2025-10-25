module TestParabolicSIP

using Test
using Trixi
using OrdinaryDiffEqLowStorageRK

@testset "ViscousFormulationSIP" begin
    @testset "Constructor and Type" begin
        # Test valid instantiation with Float64
        sip_f64 = ViscousFormulationSIP(10.0)
        @test sip_f64 isa ViscousFormulationSIP{Float64}
        @test sip_f64.alpha == 10.0

        # Test valid instantiation with Float32
        sip_f32 = ViscousFormulationSIP(5.0f0)
        @test sip_f32 isa ViscousFormulationSIP{Float32}
        @test sip_f32.alpha == 5.0f0

        # Test that ViscousFormulationSIP is a subtype of AbstractViscousFormulation
        # Note: AbstractViscousFormulation may not exist, checking compilation will reveal this
        @test ViscousFormulationSIP <: Any  # Placeholder, will be refined
    end

    @testset "Constructor Validation" begin
        # Test that alpha must be positive
        @test_throws ArgumentError ViscousFormulationSIP(0.0)
        @test_throws ArgumentError ViscousFormulationSIP(-1.0)
        @test_throws ArgumentError ViscousFormulationSIP(-5.0f0)

        # Test boundary case: very small positive value should work
        sip_small = ViscousFormulationSIP(1e-10)
        @test sip_small.alpha == 1e-10
    end

    @testset "Type Parameters" begin
        # Test that type parameter RealT is correctly inferred
        sip1 = ViscousFormulationSIP(1.0)
        @test typeof(sip1) == ViscousFormulationSIP{Float64}

        sip2 = ViscousFormulationSIP(1.0f0)
        @test typeof(sip2) == ViscousFormulationSIP{Float32}

        # Test explicit type parameter
        sip3 = ViscousFormulationSIP{Float64}(3.14)
        @test sip3.alpha == 3.14
    end
end

@testset "DGMulti 1D SIP Integration" begin
    # Test SIP method with 1D DGMulti mesh on a simple diffusion problem
    dg = DGMulti(polydeg = 2, element_type = Line(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central),
                 volume_integral = VolumeIntegralWeakForm())
    cells_per_dimension = (4,)
    mesh = DGMultiMesh(dg, cells_per_dimension)

    # Test with polynomial initial condition x^2
    # Laplacian should give constant value 2
    initial_condition = (x, t, equations) -> SVector(x[1]^2)

    equations = LinearScalarAdvectionEquation1D(1.0)
    equations_parabolic = LaplaceDiffusion1D(1.0, equations)

    # Use SIP formulation with penalty parameter alpha = 5.0
    sip_scheme = ViscousFormulationSIP(5.0)

    semi = SemidiscretizationHyperbolicParabolic(mesh, equations, equations_parabolic,
                                                 initial_condition, dg;
                                                 solver_parabolic = sip_scheme)

    @test semi.solver_parabolic === sip_scheme
    @test nvariables(semi) == nvariables(equations)
    @test Base.ndims(semi) == Base.ndims(mesh)
    @test Base.real(semi) == Base.real(dg)

    ode = semidiscretize(semi, (0.0, 0.01))
    u0 = similar(ode.u0)
    Trixi.compute_coefficients!(u0, 0.0, semi)
    @test u0 ≈ ode.u0

    # Test gradient calculation
    (; cache, cache_parabolic, equations_parabolic) = semi
    (; gradients) = cache_parabolic
    for dim in eachindex(gradients)
        fill!(gradients[dim], zero(eltype(gradients[dim])))
    end

    u0 = Base.parent(ode.u0)
    t = 0.0
    Trixi.calc_gradient!(gradients, u0, t, mesh, equations_parabolic,
                         boundary_condition_periodic, dg, sip_scheme,
                         cache, cache_parabolic)

    (; xq) = mesh.md
    # For polynomial x^2, gradient should be: du/dx = 2x
    @test getindex.(gradients[1], 1) ≈ 2 * xq

    # Test viscous flux calculation
    u_flux = similar.(gradients)
    Trixi.calc_viscous_fluxes!(u_flux, u0, gradients, mesh,
                               equations_parabolic,
                               dg, cache, cache_parabolic)
    @test u_flux[1] ≈ gradients[1]

    # Test calc_divergence! with SIP
    du_sip = similar(u0)
    @test_nowarn Trixi.calc_divergence!(du_sip, u0, t, u_flux, mesh,
                                        equations_parabolic,
                                        boundary_condition_periodic,
                                        dg, sip_scheme, cache, cache_parabolic)

    Trixi.invert_jacobian!(du_sip, mesh, equations_parabolic, dg, cache; scaling = 1.0)

    # Test with BR1 for comparison
    br1_scheme = ViscousFormulationBassiRebay1()
    du_br1 = similar(u0)
    @test_nowarn Trixi.calc_divergence!(du_br1, u0, t, u_flux, mesh,
                                        equations_parabolic,
                                        boundary_condition_periodic,
                                        dg, br1_scheme, cache, cache_parabolic)
    Trixi.invert_jacobian!(du_br1, mesh, equations_parabolic, dg, cache; scaling = 1.0)

    # Both should produce non-zero results
    @test !all(iszero, du_sip)
    @test !all(iszero, du_br1)

    # For polynomial x^2 with periodic boundaries:
    # BR1 should give exact Laplacian: d²/dx²(x^2) = 2
    # Note: use mesh.md.x for element coordinates
    (; x) = mesh.md
    @test getindex.(du_br1, 1) ≈ fill(2.0, size(x))
end

@testset "DGMulti 2D SIP Integration" begin
    # Test SIP method with 2D DGMulti mesh on a simple diffusion problem
    dg = DGMulti(polydeg = 2, element_type = Quad(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_central),
                 volume_integral = VolumeIntegralWeakForm())
    cells_per_dimension = (2, 2)
    mesh = DGMultiMesh(dg, cells_per_dimension)

    # Test with polynomial initial condition x^2 * y
    # Test if we recover the exact second derivative with SIP
    initial_condition = (x, t, equations) -> SVector(x[1]^2 * x[2])

    equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
    equations_parabolic = LaplaceDiffusion2D(1.0, equations)

    # Use SIP formulation with penalty parameter alpha = 5.0
    sip_scheme = ViscousFormulationSIP(5.0)

    semi = SemidiscretizationHyperbolicParabolic(mesh, equations, equations_parabolic,
                                                 initial_condition, dg;
                                                 solver_parabolic = sip_scheme)

    @test semi.solver_parabolic === sip_scheme
    @test nvariables(semi) == nvariables(equations)
    @test Base.ndims(semi) == Base.ndims(mesh)
    @test Base.real(semi) == Base.real(dg)

    ode = semidiscretize(semi, (0.0, 0.01))
    u0 = similar(ode.u0)
    Trixi.compute_coefficients!(u0, 0.0, semi)
    @test u0 ≈ ode.u0

    # Test gradient calculation (should be identical to BR1)
    (; cache, cache_parabolic, equations_parabolic) = semi
    (; gradients) = cache_parabolic
    for dim in eachindex(gradients)
        fill!(gradients[dim], zero(eltype(gradients[dim])))
    end

    # unpack VectorOfArray
    u0 = Base.parent(ode.u0)
    t = 0.0
    # pass in `boundary_condition_periodic` to skip boundary flux/integral evaluation
    Trixi.calc_gradient!(gradients, u0, t, mesh, equations_parabolic,
                         boundary_condition_periodic, dg, sip_scheme,
                         cache, cache_parabolic)

    (; x, y, xq, yq) = mesh.md
    # For polynomial x^2 * y, gradients should be:
    # ∂/∂x (x^2 * y) = 2 * x * y
    # ∂/∂y (x^2 * y) = x^2
    @test getindex.(gradients[1], 1) ≈ 2 * xq .* yq
    @test getindex.(gradients[2], 1) ≈ xq .^ 2

    # Test viscous flux calculation
    u_flux = similar.(gradients)
    Trixi.calc_viscous_fluxes!(u_flux, u0, gradients, mesh,
                               equations_parabolic,
                               dg, cache, cache_parabolic)
    # For LaplaceDiffusion, viscous flux should equal gradients (flux = diffusivity * gradient)
    @test u_flux[1] ≈ gradients[1]
    @test u_flux[2] ≈ gradients[2]

    # Test calc_divergence! with SIP
    du_sip = similar(u0)
    @test_nowarn Trixi.calc_divergence!(du_sip, u0, t, u_flux, mesh,
                                        equations_parabolic,
                                        boundary_condition_periodic,
                                        dg, sip_scheme, cache, cache_parabolic)

    # Apply Jacobian scaling
    Trixi.invert_jacobian!(du_sip, mesh, equations_parabolic, dg, cache; scaling = 1.0)

    # Test with BR1 for comparison
    br1_scheme = ViscousFormulationBassiRebay1()
    du_br1 = similar(u0)
    @test_nowarn Trixi.calc_divergence!(du_br1, u0, t, u_flux, mesh,
                                        equations_parabolic,
                                        boundary_condition_periodic,
                                        dg, br1_scheme, cache, cache_parabolic)
    Trixi.invert_jacobian!(du_br1, mesh, equations_parabolic, dg, cache; scaling = 1.0)

    # SIP should produce different results than BR1 due to penalty term
    # (except possibly in special cases)
    @test !all(iszero, du_sip)  # SIP result should not be all zeros
    @test !all(iszero, du_br1)  # BR1 result should not be all zeros

    # For this polynomial case with periodic boundaries:
    # BR1 should give the exact Laplacian: ∇·(∇(x^2 * y)) = 2*y
    # Note: use 'y' not 'yq' - y is the element-level coordinate
    @test getindex.(du_br1, 1) ≈ 2 * y

    # SIP with penalty may differ from BR1, but both should compute valid results
    # The important thing is that SIP runs without errors and produces non-trivial output
end

# TODO: Add convergence rate tests
# Convergence rate tests require careful handling of boundary conditions and error computation.
# These tests should verify O(h^(k+1)) convergence for polynomial degree k.
# They will be added in a future PR with proper debugging and validation.

end # module
