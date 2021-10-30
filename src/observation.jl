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
    nobs(data; obsdim = default_obsdim(data))

Return the total number of observations contained in `data`.

If it makes sense for the type of `data`, `obsdim` can be used
to indicate which dimension of `data` denotes the observations.
See [`default_obsdim`](@ref) for defining a default dimension.
"""
function nobs end

"""
    getobs(data, idx; obsdim = default_obsdim(data))

Return the observations corresponding to the observation-index `idx`.
Note that `idx` can be any type as long as `data` has defined
`getobs` for that type.

The returned observation(s) should be in the form intended to
be passed as-is to some learning algorithm. There is no strict
interface requirement on how this "actual data" must look like.
Every author behind some custom data container can make this
decision themselves.
The output should be consistent when `idx` is a scalar vs vector.

If it makes sense for the type of `data`, `obsdim` can be used
to indicate which dimension of `data` denotes the observations.
See [`default_obsdim`](@ref) for defining a default dimension.
"""
function getobs end
getobs(data, idx; obsdim = nothing) = data[idx]

"""
    getobs!(buffer, data, idx, obsdim = default_obsdim(obsdim))

Inplace version of `getobs(data, idx; obsdim)`. If this method
is defined for the type of `data`, then `buffer` should be used
to store the result, instead of allocating a dedicated object.

Implementing this function is optional. In the case no such
method is provided for the type of `data`, then `buffer` will be
*ignored* and the result of `getobs` returned. This could be
because the type of `data` may not lend itself to the concept
of `copy!`. Thus, supporting a custom `getobs!` is optional
and not required.

If it makes sense for the type of `data`, `obsdim` can be used
to indicate which dimension of `data` denotes the observations.
See [`default_obsdim`](@ref) for defining a default dimension.
"""
function getobs! end
getobs!(buffer, data, idx; obsdim = default_obsdim(data)) =
   getobs(data, idx; obsdim = obsdim)

# --------------------------------------------------------------------

# TODO: consider deprecating these

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

function targets end

# --------------------------------------------------------------------

abstract type AbstractDataContainer end

Base.getindex(x::AbstractDataContainer, i) = getobs(x, i; obsdim = default_obsdim(x))
Base.iterate(x::AbstractDataContainer, state = 1) =
    getobs(x, state; obsdim = default_obsdim(x)), state + 1

# --------------------------------------------------------------------

# Might need this distinction later
# e.g. shuffleobs can be anywhere in pipeline but
#      eachbatch is usually at the end
abstract type AbstractDataIterator <: AbstractDataContainer end


# --------------------------------------------------------------------
# Arrays

function LearnBase.nobs(A::AbstractArray; obsdim = default_obsdim(A))
    od = obsdim === nothing ? default_obsdim(A) : obsdim
    size(A, od)
end
LearnBase.nobs(A::AbstractArray{<:Any, 0}; obsdim = default_obsdim(A)) = 1

function LearnBase.getobs(A::AbstractArray{<:Any, N}, idx; obsdim = default_obsdim(A)) where N
    od = obsdim === nothing ? default_obsdim(A) : obsdim
    (od > N) && throw(BoundsError(A, (ntuple(k -> Colon(), od - 1)..., idx)))
    I = Base.setindex(map(Base.Slice, axes(A)), idx, od)
    return A[I...]
end
LearnBase.getobs(A::AbstractArray{<:Any, 0}, idx; obsdim = default_obsdim(A)) = A[idx]

function LearnBase.getobs!(buffer, A::AbstractArray, idx; obsdim = default_obsdim(obsdim))
    (obsdim > N) && throw(BoundsError(A, (ntuple(k -> Colon(), obsdim - 1)..., idx)))
    I = Base.setindex(map(Base.Slice, axes(A)), idx, obsdim)
    buffer .= A[I...]

    return buffer
end

# --------------------------------------------------------------------
# Tuples

_check_nobs_error() =
    throw(DimensionMismatch("All data containers must have the same number of observations."))

function _check_nobs(tup::Union{Tuple, NamedTuple})
    length(tup) == 0 && return
    n1 = nobs(tup[1])
    for i=2:length(tup)
        nobs(tup[i]) != n1 && _check_nobs_error()
    end
end

function _check_nobs(tup::Union{Tuple, NamedTuple}, obsdim)
    length(tup) == 0 && return
    n1 = nobs(tup[1]; obsdim = obsdim)
    for i=2:length(tup)
        nobs(tup[i]; obsdim = obsdim) != n1 && _check_nobs_error()
    end
end

function _check_nobs(tup::Union{Tuple, NamedTuple}, obsdims::Union{Tuple, NamedTuple})
    length(tup) == 0 && return
    length(tup) == length(obsdims) ||
        throw(DimensionMismatch("Number of elements in obsdim doesn't match data."))
    n1 = nobs(tup[1]; obsdim = obsdims[1])
    for i=2:length(tup)
        nobs(tup[i]; obsdim = obsdims[i]) != n1 && _check_nobs_error()
    end
end

function LearnBase.nobs(tup::Union{Tuple, NamedTuple}; obsdim = default_obsdim(tup))::Int
    _check_nobs(tup, obsdim)

    # We don't force users to handle the obsdim
    # keyword if not necessary.
    fnobs = obsdim === nothing ? nobs : x -> nobs(x; obsdim = obsdim[1])
    return length(tup) == 0 ? 0 : fnobs(tup[1])
end

LearnBase.getobs(tup::Union{Tuple, NamedTuple}, indices; obsdim = default_obsdim(tup)) =
    _getobs(tup, indices, obsdim)

function _getobs(tup::Union{Tuple, NamedTuple}, indices, obsdims::Union{Tuple, NamedTuple})
    _check_nobs(tup, obsdims)

    return map(tup, obsdims) do x, obsdim
        getobs(x, indices; obsdim = obsdim)
    end
end

function _getobs(tup::Union{Tuple, NamedTuple}, indices, obsdim)
    _check_nobs(tup, obsdim)

    return map(tup) do x
        getobs(x, indices; obsdim = obsdim)
    end
end

LearnBase.getobs!(buffers::Union{Tuple, NamedTuple}, tup::Union{Tuple, NamedTuple}, indices;
                  obsdim = default_obsdim(tup)) = 
    _getobs!(buffers, tup, indices, obsdim)

function _getobs!(buffers::Union{Tuple, NamedTuple},
                  tup::Union{Tuple, NamedTuple},
                  indices,
                  obsdims::Union{Tuple, NamedTuple})
    _check_nobs(tup, obsdims)

    return map(buffers, tup, obsdims) do buffer, x, obsdim
        getobs!(buffer, x, indices; obsdim = obsdim)
    end
end

function _getobs!(buffers::Union{Tuple, NamedTuple}, tup::Union{Tuple, NamedTuple}, indices, obsdim)
    _check_nobs(tup, obsdim)

    return map(buffers, tup) do buffer, x
        getobs!(buffer, x, indices; obsdim = obsdim)
    end
end
