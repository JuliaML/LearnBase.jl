"""
    default_obsdim(data)

Specify the default observation dimension for `data`.
Defaults to `nothing` when an observation dimension is undefined.

The following default implementations are provided:
- `default_obsdim(A::AbstractArray) = ndims(A)`
- `default_obsdim(tup::Tuple) = map(default_obsdim, tup)`
"""
default_obsdim(data) = nothing
default_obsdim(::Type{<:AbstractArray{<:Any, N}}) where N = N
default_obsdim(A::AbstractArray) = ndims(A)
default_obsdim(tup::Tuple) = map(default_obsdim, tup)

"""
   getobs(data, idx; obsdim = default_obsdim(data))

Return the observations corresponding to the observation-index `idx`.
Note that `idx` can be of type `Int` or `AbstractVector`.
*Both options must be supported by a custom type.*

The returned observation(s) should be in the form intended to
be passed as-is to some learning algorithm. There is no strict
interface requirement on how this "actual data" must look like.
Every author behind some custom data container can make this
decision himself/herself. We do, however, expect it to be consistent
for `idx` being an integer, as well as `idx` being an abstract
vector, respectively.

If it makes sense for the type of `data`, `obsdim` can be used
to indicate which dimension of `data` denotes the observations.
See [`default_obsdim`](@ref) for defining a default dimension.
"""
function getobs end
getobs(data, idx; obsdim) = getobs(data, idx)

"""
   getobs!(buffer, data, idx; obsdim = default_obsdim(obsdim))

Inplace version of `getobs(data, idx; obsdim)`. If this method
is defined for the type of `data`, then `buffer` should be used
to store the result, instead of allocating a dedicated object.

Implementing this function is optional. In the case no such
method is provided for the type of `data`, then `buffer` will be
*ignored* and the result of `getobs` returned. This could be
because the type of `data` may not lend itself to the concept
of `copy!`. Thus, supporting a custom `getobs!(::MyType, ...)`
is optional and not required.

If it makes sense for the type of `data`, `obsdim` can be used
to indicate which dimension of `data` denotes the observations.
See [`default_obsdim`](@ref) for defining a default dimension.
"""
function getobs! end
getobs!(buffer, data, idx; obsdim = default_obsdim(data)) = getobs(data, idx; obsdim = obsdim)

# --------------------------------------------------------------------

"""
   gettarget([f], observation)

Use `f` (if provided) to extract the target from the single `observation` and return it.
It is used internally by [`targets`](@ref) (only if `f` is provided)
and by [`eachtarget`](@ref) (always) on each individual observation.
"""
function gettarget end

"""
   gettargets(data, idx; obsdim = default_obsdim(data))

Return the targets corresponding to the observation-index `idx`.
Note that `idx` can be of type `Int` or `AbstractVector`.

Implementing this function for a custom type of `data` is
optional. It is particularly useful if the targets in `data` can
be provided without invoking [`getobs`](@ref). For example if you have a
remote data-source where the labels are part of some metadata
that is locally available.

If it makes sense for the type of `data`, `obsdim` can be used
to indicate which dimension of `data` denotes the observations.
See [`default_obsdim`](@ref) for defining a default dimension.
"""
function gettargets end

"""
   targets([f], data; obsdim = default_obsdim)

???
"""
function targets end

# --------------------------------------------------------------------

"""
    abstract DataView{TElem, TData} <: AbstractVector{TElem}

Baseclass for all vector-like views of some data structure.
This allow for example to see some design matrix as a vector of
individual observation-vectors instead of one matrix.

see `MLDataPattern.ObsView` and `MLDataPattern.BatchView` for examples.
"""
abstract type DataView{TElem, TData} <: AbstractVector{TElem} end

"""
    abstract AbstractObsView{TElem, TData} <: DataView{TElem, TData}

Baseclass for all vector-like views of some data structure,
that views it as some form or vector of observations.

see `MLDataPattern.ObsView` for a concrete example.
"""
abstract type AbstractObsView{TElem, TData} <: DataView{TElem, TData} end

"""
    abstract AbstractBatchView{TElem, TData} <: DataView{TElem, TData}

Baseclass for all vector-like views of some data structure,
that views it as some form or vector of equally sized batches.

see `MLDataPattern.BatchView` for a concrete example.
"""
abstract type AbstractBatchView{TElem, TData} <: DataView{TElem, TData} end

# --------------------------------------------------------------------

"""
    abstract DataIterator{TElem,TData}

Baseclass for all types that iterate over a `data` source
in some manner. The total number of observations may or may
not be known or defined and in general there is no contract that
`getobs` or `nobs` has to be supported by the type of `data`.
Furthermore, `length` should be used to query how many elements
the iterator can provide, while `nobs` may return the underlying
true amount of observations available (if known).

see `MLDataPattern.RandomObs`, `MLDataPattern.RandomBatches`
"""
abstract type DataIterator{TElem,TData} end

"""
    abstract ObsIterator{TElem,TData} <: DataIterator{TElem,TData}

Baseclass for all types that iterate over some data source
one observation at a time.

```julia
using MLDataPattern
@assert typeof(RandomObs(X)) <: ObsIterator

for x in RandomObs(X)
    # ...
end
```

see `MLDataPattern.RandomObs`
"""
abstract type ObsIterator{TElem,TData} <: DataIterator{TElem,TData} end

"""
    abstract BatchIterator{TElem,TData} <: DataIterator{TElem,TData}

Baseclass for all types that iterate over of some data source one
batch at a time.

```julia
@assert typeof(RandomBatches(X, size=10)) <: BatchIterator

for x in RandomBatches(X, size=10)
    @assert nobs(x) == 10
    # ...
end
```

see `MLDataPattern.RandomBatches`
"""
abstract type BatchIterator{TElem,TData} <: DataIterator{TElem,TData} end

# --------------------------------------------------------------------

# just for dispatch for those who care to
const AbstractDataIterator{E,T}  = Union{DataIterator{E,T}, DataView{E,T}}
const AbstractObsIterator{E,T}   = Union{ObsIterator{E,T},  AbstractObsView{E,T}}
const AbstractBatchIterator{E,T} = Union{BatchIterator{E,T},AbstractBatchView{E,T}}