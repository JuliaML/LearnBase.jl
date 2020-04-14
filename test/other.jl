@test LearnBase.Minimizable <: Any
@test LearnBase.Transformation <: Any
@test LearnBase.StochasticTransformation <: LearnBase.Transformation

@test typeof(LearnBase.transform) <: Function
@test typeof(LearnBase.transform!) <: Function
@test typeof(LearnBase.getobs) <: Function
@test typeof(LearnBase.getobs!) <: Function
@test typeof(LearnBase.learn) <: Function
@test typeof(LearnBase.learn!) <: Function
@test typeof(LearnBase.update) <: Function
@test typeof(LearnBase.update!) <: Function

@test typeof(LearnBase.grad) <: Function
@test typeof(LearnBase.grad!) <: Function

@test typeof(LearnBase.datasubset) <: Function

@test typeof(LearnBase.targets) <: Function
@test typeof(LearnBase.gettarget) <: Function
@test typeof(LearnBase.gettargets) <: Function

@test LearnBase.DataView <: AbstractVector
@test LearnBase.DataView <: LearnBase.AbstractDataIterator
@test LearnBase.DataView{Int} <: AbstractVector{Int}
@test LearnBase.DataView{Int,Vector{Int}} <: LearnBase.AbstractDataIterator{Int,Vector{Int}}
@test LearnBase.AbstractObsView <: LearnBase.DataView
@test LearnBase.AbstractObsView <: LearnBase.AbstractObsIterator
@test LearnBase.AbstractObsView{Int,Vector{Int}} <: LearnBase.DataView{Int,Vector{Int}}
@test LearnBase.AbstractObsView{Int,Vector{Int}} <: LearnBase.AbstractObsIterator{Int,Vector{Int}}
@test LearnBase.AbstractBatchView <: LearnBase.DataView
@test LearnBase.AbstractBatchView <: LearnBase.AbstractBatchIterator
@test LearnBase.AbstractBatchView{Int,Vector{Int}} <: LearnBase.DataView{Int,Vector{Int}}
@test LearnBase.AbstractBatchView{Int,Vector{Int}} <: LearnBase.AbstractBatchIterator{Int,Vector{Int}}

@test LearnBase.DataIterator <: LearnBase.AbstractDataIterator
@test LearnBase.DataIterator{Int,Vector{Int}} <: LearnBase.AbstractDataIterator{Int,Vector{Int}}
@test LearnBase.ObsIterator <: LearnBase.DataIterator
@test LearnBase.ObsIterator <: LearnBase.AbstractObsIterator
@test LearnBase.ObsIterator{Int,Vector{Int}} <: LearnBase.DataIterator{Int,Vector{Int}}
@test LearnBase.ObsIterator{Int,Vector{Int}} <: LearnBase.AbstractObsIterator{Int,Vector{Int}}
@test LearnBase.BatchIterator <: LearnBase.DataIterator
@test LearnBase.BatchIterator <: LearnBase.AbstractBatchIterator
@test LearnBase.BatchIterator{Int,Vector{Int}} <: LearnBase.DataIterator{Int,Vector{Int}}
@test LearnBase.BatchIterator{Int,Vector{Int}} <: LearnBase.AbstractBatchIterator{Int,Vector{Int}}

@test LearnBase.ObsDim.Constant <: LearnBase.ObsDimension
@test LearnBase.ObsDim.First <: LearnBase.ObsDimension
@test LearnBase.ObsDim.Last <: LearnBase.ObsDimension
@test LearnBase.ObsDim.Undefined <: LearnBase.ObsDimension
@test typeof(LearnBase.ObsDim.Constant(2)) <: LearnBase.ObsDim.Constant{2}

# IntervalSet
let s = LearnBase.IntervalSet(-1,1)
    @test typeof(s) == LearnBase.IntervalSet{Int}
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
let s = LearnBase.IntervalSet(-1,1.0)
    @test typeof(s) == LearnBase.IntervalSet{Float64}
    @test typeof(s) <: AbstractSet
    @test 1 in s
    # @show s LearnBase.randtype(s)
    @test length(s) == 1
end

# IntervalSet{Vector}
let s = LearnBase.IntervalSet([-1.,0.], [1.,1.])
    @test typeof(s) == LearnBase.IntervalSet{Vector{Float64}}
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
let s = LearnBase.DiscreteSet([-1,1])
    @test typeof(s) == LearnBase.DiscreteSet{Vector{Int}}
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
end
let s = LearnBase.DiscreteSet([-1,1.0])
    @test typeof(s) == LearnBase.DiscreteSet{Vector{Float64}}
    @test typeof(s) <: AbstractSet
    @test typeof(rand(s)) == Float64
    @test typeof(rand(s, 2)) == Vector{Float64}
end

# TupleSet
let s = LearnBase.TupleSet(LearnBase.IntervalSet(0,1), LearnBase.DiscreteSet([0,1]))
    @test typeof(s) == LearnBase.TupleSet{Tuple{LearnBase.IntervalSet{Int}, LearnBase.DiscreteSet{Vector{Int}}}}
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

    tot = 0
    for (i,x) in enumerate(s)
        @test x == s.sets[i]
        tot += length(x)
    end
    @test length(s) == tot
end

# arrays of sets
let s = [LearnBase.IntervalSet(0,1), LearnBase.DiscreteSet([0,1])]
    @test typeof(s) == Vector{AbstractSet}
    for x in ([0,0], [0.0,0.0], [0.5,1.0])
        @test x in s
    end
    for x in ([0,0.5], [-1,0])
        @test !(x in s)
    end
    @test typeof(rand(s)) == Vector{Float64}
    @test typeof(rand(s, 2)) == Vector{Vector{Float64}}
end

@testset "obsdim typetree and Constructor" begin
    @test_throws MethodError LearnBase.ObsDim.Constant(2.0)

    @test typeof(LearnBase.ObsDim.First()) <: LearnBase.ObsDimension
    @test typeof(LearnBase.ObsDim.First()) <: LearnBase.ObsDim.First
    @test typeof(LearnBase.ObsDim.First()) <: LearnBase.ObsDim.Constant{1}

    @test typeof(LearnBase.ObsDim.Last()) <: LearnBase.ObsDimension
    @test typeof(LearnBase.ObsDim.Last()) <: LearnBase.ObsDim.Last

    @test typeof(LearnBase.ObsDim.Constant(2)) <: LearnBase.ObsDimension
    @test typeof(LearnBase.ObsDim.Constant(2)) <: LearnBase.ObsDim.Constant{2}
end

@testset "obs_dim helper constructor" begin
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

struct SomeType end
@testset "obsdim default values" begin
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
