struct SomeType end
@testset "ObsDim" begin
    @testset "Type tree" begin
        @test_throws MethodError LearnBase.ObsDim.Constant(2.0)

        @test typeof(LearnBase.ObsDim.First()) <: LearnBase.ObsDimension
        @test typeof(LearnBase.ObsDim.First()) <: LearnBase.ObsDim.First
        @test typeof(LearnBase.ObsDim.First()) <: LearnBase.ObsDim.Constant{1}

        @test typeof(LearnBase.ObsDim.Last()) <: LearnBase.ObsDimension
        @test typeof(LearnBase.ObsDim.Last()) <: LearnBase.ObsDim.Last

        @test typeof(LearnBase.ObsDim.Constant(2)) <: LearnBase.ObsDimension
        @test typeof(LearnBase.ObsDim.Constant(2)) <: LearnBase.ObsDim.Constant{2}
    end

    @testset "Constructors" begin
        @test_throws ArgumentError convert(LearnBase.ObsDimension, "test")
        @test_throws ArgumentError convert(LearnBase.ObsDimension, 1.0)

        @test @inferred(convert(LearnBase.ObsDimension, LearnBase.ObsDim.First())) === LearnBase.ObsDim.First()
        @test @inferred(convert(LearnBase.ObsDimension, LearnBase.ObsDim.First())) === LearnBase.ObsDim.Constant(1)
        @test @inferred(convert(LearnBase.ObsDimension, LearnBase.ObsDim.Last()))  === LearnBase.ObsDim.Last()
        @test @inferred(convert(LearnBase.ObsDimension, LearnBase.ObsDim.Constant(2))) === LearnBase.ObsDim.Constant(2)

        @test_throws ErrorException @inferred convert(LearnBase.ObsDimension, 1)
        @test_throws ErrorException @inferred convert(LearnBase.ObsDimension, 6)
        @test convert(LearnBase.ObsDimension, 1) === LearnBase.ObsDim.First()
        @test convert(LearnBase.ObsDimension, 2) === LearnBase.ObsDim.Constant(2)
        @test convert(LearnBase.ObsDimension, 6) === LearnBase.ObsDim.Constant(6)
        @test_throws ErrorException @inferred convert(LearnBase.ObsDimension, :first)
        @test_throws ErrorException @inferred convert(LearnBase.ObsDimension, "first")
        @test convert(LearnBase.ObsDimension, (:first,:last))  === (LearnBase.ObsDim.First(),LearnBase.ObsDim.Last())
        @test convert(LearnBase.ObsDimension, :first)  === LearnBase.ObsDim.First()
        @test convert(LearnBase.ObsDimension, :begin)  === LearnBase.ObsDim.First()
        @test convert(LearnBase.ObsDimension, "first") === LearnBase.ObsDim.First()
        @test convert(LearnBase.ObsDimension, "BEGIN") === LearnBase.ObsDim.First()
        @test convert(LearnBase.ObsDimension, :end)   === LearnBase.ObsDim.Last()
        @test convert(LearnBase.ObsDimension, :last)  === LearnBase.ObsDim.Last()
        @test convert(LearnBase.ObsDimension, "End")  === LearnBase.ObsDim.Last()
        @test convert(LearnBase.ObsDimension, "LAST") === LearnBase.ObsDim.Last()
        @test convert(LearnBase.ObsDimension, :nothing) === LearnBase.ObsDim.Undefined()
        @test convert(LearnBase.ObsDimension, :none) === LearnBase.ObsDim.Undefined()
        @test convert(LearnBase.ObsDimension, :na) === LearnBase.ObsDim.Undefined()
        @test convert(LearnBase.ObsDimension, :null) === LearnBase.ObsDim.Undefined()
        @test convert(LearnBase.ObsDimension, :undefined) === LearnBase.ObsDim.Undefined()
        @test convert(LearnBase.ObsDimension, nothing) === LearnBase.ObsDim.Undefined()
    end

    @testset "Default values" begin
        @testset "Arrays, SubArrays, and Sparse Arrays" begin
            @test @inferred(LearnBase.default_obsdim(rand(10))) === LearnBase.ObsDim.Last()
            @test @inferred(LearnBase.default_obsdim(view(rand(10),:))) === LearnBase.ObsDim.Last()
            @test @inferred(LearnBase.default_obsdim(rand(10,5))) === LearnBase.ObsDim.Last()
            @test @inferred(LearnBase.default_obsdim(view(rand(10,5),:,:))) === LearnBase.ObsDim.Last()
            @test @inferred(LearnBase.default_obsdim(sprand(10,0.5))) === LearnBase.ObsDim.Last()
            @test @inferred(LearnBase.default_obsdim(sprand(10,5,0.5))) === LearnBase.ObsDim.Last()
        end

        @testset "Types with no specified default" begin
            @test @inferred(LearnBase.default_obsdim(SomeType())) === LearnBase.ObsDim.Undefined()
        end

        @testset "Tuples" begin
            @test @inferred(LearnBase.default_obsdim((SomeType(),SomeType()))) === (LearnBase.ObsDim.Undefined(), LearnBase.ObsDim.Undefined())
            @test @inferred(LearnBase.default_obsdim((SomeType(),rand(2,2)))) === (LearnBase.ObsDim.Undefined(), LearnBase.ObsDim.Last())
            @test @inferred(LearnBase.default_obsdim((rand(10),SomeType()))) === (LearnBase.ObsDim.Last(), LearnBase.ObsDim.Undefined())
            @test @inferred(LearnBase.default_obsdim((rand(10),rand(2,2)))) === (LearnBase.ObsDim.Last(), LearnBase.ObsDim.Last())
        end
    end
end
