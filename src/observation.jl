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
   getobs(data, idx, obsdim = default_obsdim(data))

Return the observations corresponding to the observation-index `idx`.
Note that `idx` can be of type `Int` or `AbstractVector`.
*Both options must be supported by a custom type.*

The returned observation(s) should be in the form intended to
be passed as-is to some learning algorithm. There is no strict
interface requirement on how this "actual data" must look like.
Every author behind some custom data container can make this
decision themselves. We do, however, expect it to be consistent
for `idx` being an integer, as well as `idx` being an abstract
vector, respectively.

If it makes sense for the type of `data`, `obsdim` can be used
to indicate which dimension of `data` denotes the observations.
See [`default_obsdim`](@ref) for defining a default dimension.
"""
function getobs end
getobs(data, idx) = data[idx]
getobs(data, idx, obsdim) = getobs(data, idx)

"""
   getobs!(buffer, data, idx, obsdim = default_obsdim(obsdim))

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
getobs!(buffer, data, idx, obsdim = default_obsdim(data)) = getobs(data, idx, obsdim)

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

abstract type AbstractDataContainer end

Base.getindex(x::AbstractDataContainer, i) = getobs(x, i, default_obsdim(x))
Base.iterate(x::AbstractDataContainer, state = 1) = getobs(x, state, default_obsdim(x)), state + 1

# --------------------------------------------------------------------

# Might need this distinction later
# e.g. shuffleobs can be anywhere in pipeline but
#      eachbatch is usually at the end
abstract type AbstractDataIterator <: AbstractDataContainer end
