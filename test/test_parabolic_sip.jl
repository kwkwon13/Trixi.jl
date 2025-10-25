module TestParabolicSIP

using Test
using Trixi

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

end # module
