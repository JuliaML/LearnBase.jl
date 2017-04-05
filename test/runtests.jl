using LearnBase
using Base.Test

# Test if types are exported properly
@test Cost <: Any
@test Penalty <: Cost
@test Loss <: Cost
@test UnsupervisedLoss <: Loss
@test SupervisedLoss <: Loss
@test MarginLoss <: SupervisedLoss
@test DistanceLoss <: SupervisedLoss

@test Minimizable <: Any
@test Transformation <: Any
@test StochasticTransformation <: Transformation

# Test if functions are exported properly
@test typeof(transform) <: Function
@test typeof(transform!) <: Function
@test typeof(getobs) <: Function
@test typeof(getobs!) <: Function
@test typeof(learn) <: Function
@test typeof(learn!) <: Function
@test typeof(update) <: Function
@test typeof(update!) <: Function

@test typeof(scaled) <: Function

@test typeof(value) <: Function
@test typeof(value!) <: Function
@test typeof(meanvalue) <: Function
@test typeof(sumvalue) <: Function
@test typeof(meanderiv) <: Function
@test typeof(sumderiv) <: Function
@test typeof(deriv) <: Function
@test typeof(deriv2) <: Function
@test typeof(deriv!) <: Function
@test typeof(value_deriv) <: Function
@test typeof(grad) <: Function
@test typeof(grad!) <: Function
@test typeof(addgrad!) <: Function
@test typeof(value_grad) <: Function
@test typeof(value_grad!) <: Function
@test typeof(prox) <: Function
@test typeof(prox!) <: Function

@test typeof(isminimizable) <: Function
@test typeof(isdifferentiable) <: Function
@test typeof(istwicedifferentiable) <: Function
@test typeof(isconvex) <: Function
@test typeof(isstrictlyconvex) <: Function
@test typeof(isstronglyconvex) <: Function
@test typeof(isnemitski) <: Function
@test typeof(isunivfishercons) <: Function
@test typeof(isfishercons) <: Function
@test typeof(islipschitzcont) <: Function
@test typeof(islocallylipschitzcont) <: Function
@test typeof(islipschitzcont_deriv) <: Function
@test typeof(isclipable) <: Function
@test typeof(ismarginbased) <: Function
@test typeof(isclasscalibrated) <: Function
@test typeof(isdistancebased) <: Function

@test typeof(issymmetric) <: Function

@test typeof(getobs) <: Function
@test typeof(getobs!) <: Function
@test typeof(datasubset) <: Function

@test typeof(targets) <: Function
@test typeof(LearnBase.gettarget) <: Function
@test typeof(LearnBase.gettargets) <: Function

@test DataView <: AbstractVector
@test DataView <: AbstractDataIterator
@test DataView{Int} <: AbstractVector{Int}
@test DataView{Int,Vector{Int}} <: AbstractDataIterator{Int,Vector{Int}}
@test AbstractObsView <: DataView
@test AbstractObsView <: AbstractObsIterator
@test AbstractObsView{Int,Vector{Int}} <: DataView{Int,Vector{Int}}
@test AbstractObsView{Int,Vector{Int}} <: AbstractObsIterator{Int,Vector{Int}}
@test AbstractBatchView <: DataView
@test AbstractBatchView <: AbstractBatchIterator
@test AbstractBatchView{Int,Vector{Int}} <: DataView{Int,Vector{Int}}
@test AbstractBatchView{Int,Vector{Int}} <: AbstractBatchIterator{Int,Vector{Int}}

@test DataIterator <: AbstractDataIterator
@test DataIterator{Int,Vector{Int}} <: AbstractDataIterator{Int,Vector{Int}}
@test ObsIterator <: DataIterator
@test ObsIterator <: AbstractObsIterator
@test ObsIterator{Int,Vector{Int}} <: DataIterator{Int,Vector{Int}}
@test ObsIterator{Int,Vector{Int}} <: AbstractObsIterator{Int,Vector{Int}}
@test BatchIterator <: DataIterator
@test BatchIterator <: AbstractBatchIterator
@test BatchIterator{Int,Vector{Int}} <: DataIterator{Int,Vector{Int}}
@test BatchIterator{Int,Vector{Int}} <: AbstractBatchIterator{Int,Vector{Int}}

@test typeof(fit) <: Function
@test typeof(fit!) <: Function
@test typeof(nobs) <: Function

@test ObsDim.Constant <: LearnBase.ObsDimension
@test ObsDim.First <: LearnBase.ObsDimension
@test ObsDim.Last <: LearnBase.ObsDimension
@test ObsDim.Undefined <: LearnBase.ObsDimension
@test typeof(ObsDim.Constant(2)) <: ObsDim.Constant{2}

# Test that LearnBase reuses StatsBase functions
using StatsBase
@test StatsBase.fit  == LearnBase.fit
@test StatsBase.fit! == LearnBase.fit!
@test StatsBase.nobs == LearnBase.nobs

# Test superset fallbacks
immutable MyStronglyConvexType end
LearnBase.isstronglyconvex(::MyStronglyConvexType) = true
LearnBase.islipschitzcont(::MyStronglyConvexType) = true
@test isstronglyconvex(MyStronglyConvexType())
@test isstrictlyconvex(MyStronglyConvexType())
@test isconvex(MyStronglyConvexType())
@test islipschitzcont(MyStronglyConvexType())
@test islocallylipschitzcont(MyStronglyConvexType())

# IntervalSet
let s = IntervalSet(-1,1)
    @test typeof(s) == IntervalSet{Int}
    @test typeof(s) <: AbstractSet
    for x in (-1,0,0.5,1,1.0)
        @test x in s
    end
    for x in (-1-1e-10, 1+1e-10, -Inf, Inf, 2, NaN)
        @test !(x in s)
    end
    for i=1:10
        x = rand(s)
        @test typeof(x) == Float64
        @test x in s
    end
    xs = rand(s, 10)
    @test typeof(xs) == Vector{Float64}
    for x in xs
        @test typeof(x) == Float64
        @test x in s
    end
    @test LearnBase.randtype(s) == Float64
    # @show s LearnBase.randtype(s)
end
let s = IntervalSet(-1,1.0)
    @test typeof(s) == IntervalSet{Float64}
    @test typeof(s) <: AbstractSet
    @test 1 in s
    # @show s LearnBase.randtype(s)
    @test length(s) == 1
end

# IntervalSet{Vector}
let s = IntervalSet([-1.,0.], [1.,1.])
    @test typeof(s) == IntervalSet{Vector{Float64}}
    @test typeof(s) <: AbstractSet
    @test LearnBase.randtype(s) == Vector{Float64}
    @test typeof(rand(s)) == Vector{Float64}
    @test rand(s) in s
    @test [-1, 0] in s
    @test !([-1.5,0] in s)
    @test !([0,2] in s)
    @test length(s) == 2
end

# DiscreteSet
let s = DiscreteSet([-1,1])
    @test typeof(s) == DiscreteSet{Vector{Int}}
    @test typeof(s) <: AbstractSet
    for x in (-1, 1, -1.0, 1.0)
        @test x in s
    end
    for x in (0, Inf, -Inf, NaN)
        @test !(x in s)
    end
    for i=1:10
        x = rand(s)
        @test typeof(x) == Int
        @test x in s
    end
    xs = rand(s, 10)
    @test typeof(xs) == Vector{Int}
    for x in xs
        @test typeof(x) == Int
        @test x in s
    end
    @test LearnBase.randtype(s) == Int
    @test length(s) == 2
    @test s[1] == -1
    # @show s LearnBase.randtype(s)
end
let s = DiscreteSet([-1,1.0])
    @test typeof(s) == DiscreteSet{Vector{Float64}}
    @test typeof(s) <: AbstractSet
    @test typeof(rand(s)) == Float64
    @test typeof(rand(s, 2)) == Vector{Float64}
    # @show s LearnBase.randtype(s)
end

# TupleSet
let s = TupleSet(IntervalSet(0,1), DiscreteSet([0,1]))
    @test typeof(s) == TupleSet{Tuple{IntervalSet{Int}, DiscreteSet{Vector{Int}}}}
    @test typeof(s) <: AbstractSet
    for x in ([0,0], [0.0,0.0], [0.5,1.0])
        @test x in s
    end
    for x in ([0,0.5], [-1,0])
        @test !(x in s)
    end
    @test typeof(rand(s)) == Vector{Float64}
    @test typeof(rand(s, 2)) == Vector{Vector{Float64}}
    @test typeof(rand(s, Tuple)) == Tuple{Float64,Int}
    @test typeof(rand(s, Tuple, 2)) == Vector{Tuple{Float64,Int}}
    @test LearnBase.randtype(s) == Vector{Float64}
    # @show s LearnBase.randtype(s)

    tot = 0
    for (i,x) in enumerate(s)
        @test x == s.sets[i]
        tot += length(x)
    end
    @test length(s) == tot
end

# arrays of sets
let s = [IntervalSet(0,1), DiscreteSet([0,1])]
    @test typeof(s) == Vector{AbstractSet}
    for x in ([0,0], [0.0,0.0], [0.5,1.0])
        @test x in s
    end
    for x in ([0,0.5], [-1,0])
        @test !(x in s)
    end
    @test typeof(rand(s)) == Vector{Float64}
    @test typeof(rand(s, 2)) == Vector{Vector{Float64}}
    # @show s LearnBase.randtype(s)
end

@testset "obsdim typetree and Constructor" begin
    @test_throws MethodError ObsDim.Constant(2.0)

    @test typeof(ObsDim.First()) <: LearnBase.ObsDimension
    @test typeof(ObsDim.First()) <: ObsDim.First
    @test typeof(ObsDim.First()) <: ObsDim.Constant{1}

    @test typeof(ObsDim.Last()) <: LearnBase.ObsDimension
    @test typeof(ObsDim.Last()) <: ObsDim.Last

    @test typeof(ObsDim.Constant(2)) <: LearnBase.ObsDimension
    @test typeof(ObsDim.Constant(2)) <: ObsDim.Constant{2}
end

@testset "obs_dim helper constructor" begin
    @test_throws ArgumentError convert(LearnBase.ObsDimension, "test")
    @test_throws ArgumentError convert(LearnBase.ObsDimension, 1.0)

    @test @inferred(convert(LearnBase.ObsDimension, ObsDim.First())) === ObsDim.First()
    @test @inferred(convert(LearnBase.ObsDimension, ObsDim.First())) === ObsDim.Constant(1)
    @test @inferred(convert(LearnBase.ObsDimension, ObsDim.Last()))  === ObsDim.Last()
    @test @inferred(convert(LearnBase.ObsDimension, ObsDim.Constant(2))) === ObsDim.Constant(2)

    @test_throws ErrorException @inferred convert(LearnBase.ObsDimension, 1)
    @test_throws ErrorException @inferred convert(LearnBase.ObsDimension, 6)
    @test convert(LearnBase.ObsDimension, 1) === ObsDim.First()
    @test convert(LearnBase.ObsDimension, 2) === ObsDim.Constant(2)
    @test convert(LearnBase.ObsDimension, 6) === ObsDim.Constant(6)
    @test_throws ErrorException @inferred convert(LearnBase.ObsDimension, :first)
    @test_throws ErrorException @inferred convert(LearnBase.ObsDimension, "first")
    @test convert(LearnBase.ObsDimension, (:first,:last))  === (ObsDim.First(),ObsDim.Last())
    @test convert(LearnBase.ObsDimension, :first)  === ObsDim.First()
    @test convert(LearnBase.ObsDimension, :begin)  === ObsDim.First()
    @test convert(LearnBase.ObsDimension, "first") === ObsDim.First()
    @test convert(LearnBase.ObsDimension, "BEGIN") === ObsDim.First()
    @test convert(LearnBase.ObsDimension, :end)   === ObsDim.Last()
    @test convert(LearnBase.ObsDimension, :last)  === ObsDim.Last()
    @test convert(LearnBase.ObsDimension, "End")  === ObsDim.Last()
    @test convert(LearnBase.ObsDimension, "LAST") === ObsDim.Last()
    @test convert(LearnBase.ObsDimension, :nothing) === ObsDim.Undefined()
    @test convert(LearnBase.ObsDimension, :none) === ObsDim.Undefined()
    @test convert(LearnBase.ObsDimension, :na) === ObsDim.Undefined()
    @test convert(LearnBase.ObsDimension, :null) === ObsDim.Undefined()
    @test convert(LearnBase.ObsDimension, :undefined) === ObsDim.Undefined()
    @test convert(LearnBase.ObsDimension, nothing) === ObsDim.Undefined()

    # TODO: remove test after deprecation phase
    @test LearnBase.obs_dim(:last) === ObsDim.Last()
end

immutable SomeType end
@testset "obsdim default values" begin
    @testset "Arrays, SubArrays, and Sparse Arrays" begin
        @test @inferred(LearnBase.default_obsdim(rand(10))) === ObsDim.Last()
        @test @inferred(LearnBase.default_obsdim(view(rand(10),:))) === ObsDim.Last()
        @test @inferred(LearnBase.default_obsdim(rand(10,5))) === ObsDim.Last()
        @test @inferred(LearnBase.default_obsdim(view(rand(10,5),:,:))) === ObsDim.Last()
        @test @inferred(LearnBase.default_obsdim(sprand(10,0.5))) === ObsDim.Last()
        @test @inferred(LearnBase.default_obsdim(sprand(10,5,0.5))) === ObsDim.Last()
    end

    @testset "Types with no specified default" begin
        @test @inferred(LearnBase.default_obsdim(SomeType())) === ObsDim.Undefined()
    end

    @testset "Tuples" begin
        @test @inferred(LearnBase.default_obsdim((SomeType(),SomeType()))) === (ObsDim.Undefined(), ObsDim.Undefined())
        @test @inferred(LearnBase.default_obsdim((SomeType(),rand(2,2)))) === (ObsDim.Undefined(), ObsDim.Last())
        @test @inferred(LearnBase.default_obsdim((rand(10),SomeType()))) === (ObsDim.Last(), ObsDim.Undefined())
        @test @inferred(LearnBase.default_obsdim((rand(10),rand(2,2)))) === (ObsDim.Last(), ObsDim.Last())
    end
end

