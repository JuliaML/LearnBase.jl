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

@test Minimizeable <: Any
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

@test typeof(fit) <: Function
@test typeof(fit!) <: Function
@test typeof(nobs) <: Function

# Test that LearnBase reuses StatsBase functions
using StatsBase
@test StatsBase.fit  == LearnBase.fit
@test StatsBase.fit! == LearnBase.fit!
@test StatsBase.nobs == LearnBase.nobs

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
end
let s = IntervalSet(-1,1.0)
	@test typeof(s) == IntervalSet{Float64}
	@test typeof(s) <: AbstractSet
	@test 1 in s
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
end
let s = DiscreteSet([-1,1.0])
	@test typeof(s) == DiscreteSet{Vector{Float64}}
	@test typeof(s) <: AbstractSet
	@test typeof(rand(s)) == Float64
	@test typeof(rand(s,2)) == Vector{Float64}
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
	@test typeof(rand(s)) == Vector{Real}
	@test typeof(rand(s,2)) == Vector{Vector{Real}}
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
	@test typeof(rand(s)) == Vector{Real}
	@test typeof(rand(s,2)) == Vector{Vector{Real}}
end

