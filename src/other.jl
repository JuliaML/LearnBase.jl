"""
Return the gradient of the learnable parameters w.r.t. some objective
"""
function grad end
function grad! end

"""
Proximal operator of a function (https://en.wikipedia.org/wiki/Proximal_operator)
"""
function prox end
function prox! end

"""
Anything that takes an input and performs some kind
of function to produce an output. For example a linear
prediction function.
"""
abstract type Transformation end
abstract type StochasticTransformation <: Transformation end
abstract type Learnable <: Transformation end

"""
Do a forward pass, and return the output
"""
function transform end
function transform! end

"""
Baseclass for any prediction model that can be minimized.
This means that an object of a subclass contains all the
information needed to compute its own current loss.
"""
abstract type Minimizable <: Learnable end

function update end
function update! end
function learn end
function learn! end

# --------------------------------------------------------------------

import Base: AbstractSet

"A continuous range (inclusive) between a lo and a hi"
struct IntervalSet{T} <: AbstractSet{T}
    lo::T
    hi::T
end
function IntervalSet(lo::A, hi::B) where {A,B}
    T = promote_type(A,B)
    IntervalSet{T}(convert(T,lo), convert(T,hi))
end

# numeric interval
randtype(s::IntervalSet{T}) where T <: Number = Float64
Base.rand(s::IntervalSet{T}, dims::Integer...) where T <: Number = rand(dims...) .* (s.hi - s.lo) .+ s.lo
Base.in(x::Number, s::IntervalSet{T}) where T <: Number = s.lo <= x <= s.hi
Base.length(s::IntervalSet{T}) where T <: Number = 1
Base.:(==)(s1::IntervalSet{T}, s2::IntervalSet{T}) where T = s1.lo == s2.lo && s1.hi == s2.hi


# vector of intervals
randtype(s::IntervalSet{T}) where T <: AbstractVector = Vector{Float64}
Base.rand(s::IntervalSet{T}) where T <: AbstractVector = Float64[rand() * (s.hi[i] - s.lo[i]) + s.lo[i] for i=1:length(s)]
Base.in(x::AbstractVector, s::IntervalSet{T}) where T <: AbstractVector = all(i -> s.lo[i] <= x[i] <= s.hi[i], 1:length(s))
Base.length(s::IntervalSet{T}) where T <: AbstractVector = length(s.lo)


"Set of discrete items"
struct DiscreteSet{T<:AbstractArray} <: AbstractSet{T}
    items::T
end
randtype(s::DiscreteSet) = eltype(s.items)
Base.rand(s::DiscreteSet, dims::Integer...) = rand(s.items, dims...)
Base.in(x, s::DiscreteSet) = x in s.items
Base.length(s::DiscreteSet) = length(s.items)
Base.getindex(s::DiscreteSet, i::Int) = s.items[i]
Base.:(==)(s1::DiscreteSet, s2::DiscreteSet) = s1.items == s2.items


# operations on arrays of sets
randtype(sets::AbstractArray{S,N}) where {S <: AbstractSet, N} = Array{promote_type(map(randtype, sets)...), N}
Base.rand(sets::AbstractArray{S}) where S <: AbstractSet = eltype(randtype(sets))[rand(s) for s in sets]
function Base.rand(sets::AbstractArray{S}, dim1::Integer, dims::Integer...) where S <: AbstractSet
    A = Array{randtype(sets)}(undef, dim1, dims...)
    for i in eachindex(A)
        A[i] = rand(sets)
    end
    A
end
function Base.in(xs::AbstractArray, sets::AbstractArray{S}) where S <: AbstractSet
    size(xs) == size(sets) && all(map(in, xs, sets))
end


"Groups several heterogenous sets. Used mainly for proper dispatch."
struct TupleSet{T<:Tuple} <: AbstractSet{T}
    sets::T
end
TupleSet(sets::AbstractSet...) = TupleSet(sets)

# rand can return arrays or tuples, but defaults to arrays
randtype(sets::TupleSet, ::Type{Vector}) = Vector{promote_type(map(randtype, sets.sets)...)}
Base.rand(sets::TupleSet, ::Type{Vector}) = eltype(randtype(sets, Vector))[rand(s) for s in sets.sets]
randtype(sets::TupleSet, ::Type{Tuple}) = Tuple{map(randtype, sets.sets)...}
Base.rand(sets::TupleSet, ::Type{Tuple}) = map(rand, sets.sets)
function Base.rand(sets::TupleSet, ::Type{OT}, dim1::Integer, dims::Integer...) where OT
    A = Array{randtype(sets, OT)}(undef, dim1, dims...)
    for i in eachindex(A)
        A[i] = rand(sets, OT)
    end
    A
end
Base.length(sets::TupleSet) = sum(length(s) for s in sets.sets)
Base.iterate(sets::TupleSet) = iterate(sets.sets)
Base.iterate(sets::TupleSet, i) = iterate(sets.sets, i)

randtype(sets::TupleSet) = randtype(sets, Vector)
Base.rand(sets::TupleSet, dims::Integer...) = rand(sets, Vector, dims...)
Base.in(x, sets::TupleSet) = all(map(in, x, sets.sets))

"Returns an AbstractSet representing valid input values"
function inputdomain end

"Returns an AbstractSet representing valid output/target values"
function targetdomain end
