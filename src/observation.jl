"""
    default_obsdim(data)

Specify the default observation dimension for `data`.
Falls back to `nothing` when an observation dimension is undefined.

By default, the following implementations are provided:
- `default_obsdim(A::AbstractArray) = ndims(A)`
- `default_obsdim(tup::Tuple) = map(default_obsdim, tup)`
"""
default_obsdim(data) = nothing
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
for `idx` being an integer, as well as `idx` being a vector, respectively.

If it makes sense for the type of `data`, `obsdim` can be used
to indicate which dimension of `data` denotes the observations.
See [`default_obsdim`](@ref) for defining a default dimension.

Data types implementing `LearnBase.getobs` typically also
implement `StatsBase.nobs`.

# Examples

Let's see how to implement a dataset interface for a dataset
represented by an array:
```julia
using LearnBase

function LearnBase.getobs(x::AbstractArray{T,N}, idx; obsdim=default_obsdim(x)) where {T,N}   
    _idx = ntuple(i->  i == obsdim ? idx : Colon(), N)
    return x[_idx...]
end

# LearnBase imports nobs from StatsBase
LearnBase.nobs(x::AbstractArray; obsdim=default_obsdim(x)) = size(x, obsdim)  

X = rand(2,3)

nobs(X) # == 3

getobs(X, 2) # same as X[:,2]
```
In a supervised learning setting, it can be convenient to 
interpret a tuple of arrays as the inputs and the tagets:
```julia
# Here we use Ref to protect idx against broadcasting
LearnBase.getobs(t::Tuple, idx) = getobs.(t, Ref(idx))

# Assume all elements have the same nummber of observations.
# It would be safer to check explicitely though.
StatsBase.nobs(t::Tuple) = nobs(t[1])

# A dataset with 3 observations, each with 2 input features
X, Y = rand(2, 3), rand(3)
dataset = (X, Y) 

getobs(dataset, 2) # -> (X[:,2], Y[2])
getobs(dataset, 1:2) # -> (X[:,1:2], Y[1:2])
```
"""
function getobs end

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
     datasubset(data, [idx]; obsdim = default_obsdim(data))

Return a lazy subset of the observations in `data` that correspond
to the given `idx`. No data should be copied except of the
indices. Note that `idx` can be of type `Int` or `AbstractVector`.
Both options must be supported by a custom type.
If it makes sense for the type of `data`, `obsdim` can be used
to disptach on which dimension of `data` denotes the observations.
"""
function datasubset end

# todeprecate
function target end
function gettarget end