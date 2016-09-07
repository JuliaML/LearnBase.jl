__precompile__(true)

module LearnBase

# Only reexport required functions by default
import StatsBase: nobs, fit, fit!

# We temporary reexport issymmetric for smooth
# transition into 0.5
if VERSION >= v"0.5-"
    import Base.issymmetric
else
    const issymmetric = Base.issym
end

"""
Baseclass for any kind of cost. Notable examples for
costs are `Loss` and `Penalty`.
"""
abstract Cost

"""
Baseclass for all losses. A loss is some (possibly simplified)
function `L(features, targets, outputs)`, where `outputs` are the
result of some function `f(features)`.
"""
abstract Loss <: Cost

"""
A loss is considered **supervised**, if all the information needed
to compute `L(features, targets, outputs)` are contained in
`targets` and `outputs`, and thus allows for the simplification
`L(targets, outputs)`.
"""
abstract SupervisedLoss <: Loss

"""
A supervised loss, where the targets are in {-1, 1}, and which
can be simplified to `L(targets, outputs) = L(targets * outputs)`
is considered **margin-based**.
"""
abstract MarginLoss <: SupervisedLoss


"""
A supervised loss that can be simplified to
`L(targets, outputs) = L(targets - outputs)` is considered
distance-based.
"""
abstract DistanceLoss <: SupervisedLoss

"""
A loss is considered **usupervised**, if all the information needed
to compute `L(features, targets, outputs)` are contained in
`features` and `outputs`, and thus allows for the simplification
`L(features, outputs)`.
"""
abstract UnsupervisedLoss <: Loss

abstract Penalty <: Cost

function value end
function value! end
function meanvalue end
function sumvalue end
function meanderiv end
function sumderiv end
function deriv end
function deriv2 end
function deriv! end
function value_deriv end
function grad end
function grad! end
function addgrad! end
function value_grad end
function value_grad! end
function prox end
function prox! end

function isminimizable end
function isdifferentiable end
function istwicedifferentiable end
function isconvex end
function isstrictlyconvex end
function isstronglyconvex end
function isnemitski end
function isunivfishercons end
function isfishercons end
function islipschitzcont end
function islocallylipschitzcont end
function islipschitzcont_deriv end # maybe overkill
function isclipable end
function ismarginbased end
function isclasscalibrated end
function isdistancebased end

# fallback to supersets
isstrictlyconvex(any) = isstronglyconvex(any)
isconvex(any) = isstrictlyconvex(any)
islocallylipschitzcont(any) = islipschitzcont(any)

"""
Anything that takes an input and performs some kind
of function to produce an output. For example a linear
prediction function.
"""
abstract Transformation
abstract StochasticTransformation <: Transformation

function transform end
function transform! end

"""
Baseclass for any prediction model that can be minimized.
This means that an object of a subclass contains all the
information needed to compute its own current loss.
"""
abstract Minimizeable <: Transformation

function getobs end
function getobs! end

function update end
function update! end
function learn end
function learn! end


"""
Base class for defining sets of numbers. Used in specifying
the input and output domains of transformations and losses.
"""
abstract AbstractSet

"A continuous range (inclusive) between a lo and a hi"
immutable IntervalSet{T<:Number} <: AbstractSet
    lo::T
    hi::T
end
function IntervalSet{A<:Number,B<:Number}(lo::A, hi::B)
    T = promote_type(A,B)
    IntervalSet{T}(convert(T,lo), convert(T,hi))
end
randtype(s::IntervalSet) = Float64
Base.rand{T<:Number}(s::IntervalSet{T}, dims::Integer...) = rand(dims...) * (s.hi - s.lo) + s.lo
Base.in{T<:Number}(x::Number, s::IntervalSet{T}) = s.lo <= x <= s.hi


"Set of discrete items"
immutable DiscreteSet{T<:AbstractArray} <: AbstractSet
    items::T
end
randtype(s::DiscreteSet) = eltype(s.items)
Base.rand(s::DiscreteSet, dims::Integer...) = rand(s.items, dims...)
Base.in(x, s::DiscreteSet) = x in s.items
Base.length(s::DiscreteSet) = length(s.items)
Base.getindex(s::DiscreteSet, i::Int) = s.items[i]


# operations on arrays of sets
randtype{S<:AbstractSet,N}(sets::AbstractArray{S,N}) = Array{promote_type(map(randtype, sets)...), N}
Base.rand{S<:AbstractSet}(sets::AbstractArray{S}) = eltype(randtype(sets))[rand(s) for s in sets]
function Base.rand{S<:AbstractSet}(sets::AbstractArray{S}, dim1::Integer, dims::Integer...)
    A = Array(randtype(sets), dim1, dims...)
    for i in eachindex(A)
        A[i] = rand(sets)
    end
    A
end
function Base.in{S<:AbstractSet}(xs::AbstractArray, sets::AbstractArray{S})
    size(xs) == size(sets) && all(map(in, xs, sets))
end


"Groups several heterogenous sets. Used mainly for proper dispatch."
immutable TupleSet{T<:Tuple} <: AbstractSet
    sets::T
end
TupleSet(sets::AbstractSet...) = TupleSet(sets)

# rand can return arrays or tuples, but defaults to arrays
randtype(sets::TupleSet, ::Type{Vector}) = Vector{promote_type(map(randtype, sets.sets)...)}
Base.rand(sets::TupleSet, ::Type{Vector}) = eltype(randtype(sets, Vector))[rand(s) for s in sets.sets]
randtype(sets::TupleSet, ::Type{Tuple}) = Tuple{map(randtype, sets.sets)...}
Base.rand(sets::TupleSet, ::Type{Tuple}) = map(rand, sets.sets)
function Base.rand{OT}(sets::TupleSet, ::Type{OT}, dim1::Integer, dims::Integer...)
    A = Array(randtype(sets, OT), dim1, dims...)
    for i in eachindex(A)
        A[i] = rand(sets, OT)
    end
    A
end

randtype(sets::TupleSet) = randtype(sets, Vector)
Base.rand(sets::TupleSet, dims::Integer...) = rand(sets, Vector, dims...)
Base.in(x, sets::TupleSet) = all(map(in, x, sets.sets))

"Returns an AbstractSet representing valid input values"
function inputdomain end

"Returns an AbstractSet representing valid output/target values"
function targetdomain end


export

    # Types
    Cost,
        Loss,
            SupervisedLoss,
                MarginLoss,
                DistanceLoss,
            UnsupervisedLoss,
        Penalty,

    Transformation,
        StochasticTransformation,

    Minimizeable,

    AbstractSet,
        IntervalSet,
        DiscreteSet,
        TupleSet,

    # Functions
    getobs,
    getobs!,

    learn,
    learn!,
    update,
    update!,
    transform,
    transform!,
    value,
    value!,
    meanvalue,
    sumvalue,
    meanderiv,
    sumderiv,
    deriv,
    deriv!,
    grad,
    grad!,
    addgrad!,
    deriv2,
    value_deriv,
    value_deriv!,
    value_grad,
    value_grad!,
    prox,
    prox!,
    inputdomain,
    targetdomain,

    isminimizable,
    isdifferentiable,
    istwicedifferentiable,
    isconvex,
    isstrictlyconvex,
    isstronglyconvex,
    isnemitski,
    isunivfishercons,
    isfishercons,
    islipschitzcont,
    islocallylipschitzcont,
    islipschitzcont_deriv,
    isclipable,
    ismarginbased,
    isclasscalibrated,
    isdistancebased,

    # Base
    issymmetric,

    # StatsBase
    fit,
    fit!,
    nobs

end # module
