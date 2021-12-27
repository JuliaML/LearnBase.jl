# T = Eltype, K = Labelcount, M = Arraydimensions
abstract type LabelEncoding{T,K,M} end
# act as scalar in broadcast, see julia #18618
Base.Broadcast.broadcastable(x::LabelEncoding) = Ref(x)

const BinaryLabelEncoding{T,M} = LabelEncoding{T,2,M}
const VectorLabelEncoding{T,K} = LabelEncoding{T,K,1}
const MatrixLabelEncoding{T,K} = LabelEncoding{T,K,2}

"""
    nlabel(obj)::Int

Returns the number of labels represented in the given object `obj`.
"""
function nlabel end

"""
    label(obj)::Vector

Returns the labels represented in the given object `obj`.
Note that the order of the labels matters.
In the case of two labels, the first element represents the positive
label and the second element the negative label.
"""
function label end

"""
    labeltype(::Type{<:LabelEncoding})

Determine the type of the labels represented by the label encoding.
"""
function labeltype end

"""
    ind2label(index, encoding)

Converts the given `index` into the corresponding label defined by the `encoding`.
Note that in the binary case, `index == 1` represents the positive label
and `index == 2` the negative label.
"""
function ind2label end

"""
    label2ind(label, encoding)::Int

Converts the given `label` into the corresponding index defined by the encoding.
Note that in the binary case, the positive label
will result in the index `1` and the negative label in the index `2`, respectively.
"""
function label2ind end

"""
    poslabel(encoding)

If the encoding is binary it will return the positive label of it.
The function will throw an error otherwise.
"""
function poslabel end

"""
    neglabel(encoding)

If the encoding is binary it will return the negative label of it.
The function will throw an error otherwise.
"""
function neglabel end

"""
    labelenc(obj) -> LabelEncoding

Tries to determine the most approriate label-encoding to describe the
given object `obj` based on the result of `label(obj)`. Note that in
most cases this function is not typestable.

```juliarepl
julia> labelenc([:yes,:no,:no,:yes,:maybe])
MLLabelUtils.LabelEnc.NativeLabels{Symbol,3}(Symbol[:yes,:no,:maybe],Dict(:yes=>1,:maybe=>3,:no=>2))

julia> labelenc([1,0,0,1,0,1])
MLLabelUtils.LabelEnc.ZeroOne{Int64,Float64}(0.5)

julia> labelenc(Int8[1,-1,-1,1,-1,1])
MLLabelUtils.LabelEnc.MarginBased{Int8}()
```
"""
function labelenc end

"""
    islabelenc(obj, encoding)

Checks is the given object `obj` can be described as being produced
by the given `encoding` in which case the function returns true, or false otherwise.
"""
function islabelenc end

"""
    isposlabel(x, encoding)

Checks if the given value `x` can be interpreted as the positive label
given the `encoding`. This function takes potential classification rules into account.
"""
function isposlabel end

"""
    isneglabel(x, encoding)

Checks if the given value `x` can be interpreted as the negative label
given the `encoding`. This function takes potential classification rules into account.
"""
function isneglabel end

"""
    classify(x, encoding)

Returns the classified version of `x` given the `encoding`.
Which means that if `x` can be interpreted as a positive label,
the positive label of `encoding` is returned; the negative otherwise.
"""
function classify end

"""
    classify!(out, x, encoding)

Same as `classify`, but uses `out` to store the result.
"""
function classify! end

"""
    convertlabel(new_encoding, x, [old_encoding])

Converts the given value/array `x` from the `old_encoding` into the
`new_encoding`. Note that if `old_encoding` is not specified it will
be derived automaticaly using `labelenc`.

```juliarepl
julia> convertlabel(LabelEnc.MarginBased, [0, 1, 1, 0, 0])
5-element Array{Int64,1}:
    -1
    1
    1
    -1
    -1

julia> convertlabel([:yes,:no], [0, 1, 1, 0, 0])
5-element Array{Symbol,1}:
    :no
    :yes
    :yes
    :no
    :no
```

For more information on the available encodings, see `?LabelEnc`.

    convertlabel(new_encoding, x, [old_encoding], [obsdim])

When working with `OneOfK` one can additionally specifify which
dimension of the array denotes the observations using `obsdim`

```juliarepl
julia> convertlabel(LabelEnc.OneOfK, [0, 1, 1, 0, 0], obsdim = 2)
2×5 Array{Int64,2}:
    0  1  1  0  0
    1  0  0  1  1
```
"""
function convertlabel end
function convertlabel! end

"""
    convertlabel(new_encoding, vec::AbstractVector, [old_encoding]) -> (Readonly)MappedArray

Creates a lazy view into `vec` that makes it look like it is
in the encoding specified by `new_encoding`, while it is actually
preserved as being of `old_encoding`.

This method only works for label-encodings that are vector-based
(i.e. pretty much all but `OneOfK`). The resulting MappedArray
will be writeable unless `old_encoding` is of type `OneVsRest`,
in which case the result will be a `ReadonlyMappedArray`.
"""
function convertlabelview end

"""
    labelmap(obj) -> Dict

Computes a mapping from the labels in `obj` to all the individual
element-indices in `obj` that correspond to that label

```juliarepl
julia> labelmap([0, 1, 1, 0, 0])
Dict{Int64,Array{Int64,1}} with 2 entries:
    0 => [1,4,5]
    1 => [2,3]
```
"""
function labelmap end

"""
    labelmap!(dict, idx, elem) -> Dict

Updates the given label-map `dict` with the new element `elem`,
which is assumed to be associated with the index `idx`.

```juliarepl
julia> lm = labelmap([0, 1, 1, 0, 0])
Dict{Int64,Array{Int64,1}} with 2 entries:
    0 => [1,4,5]
    1 => [2,3]

julia> labelmap!(lm, 6, 0)
Dict{Int64,Array{Int64,1}} with 2 entries:
    0 => [1,4,5,6]
    1 => [2,3]

julia> labelmap!(lm, 7:8, [1,0])
Dict{Int64,Array{Int64,1}} with 2 entries:
    0 => [1,4,5,6,8]
    1 => [2,3,7]
```
"""
function labelmap! end

"""
    labelfreq(obj) -> Dict

Computes the absolute frequencies for each label in `obj`.

```juliarepl
julia> labelfreq([0, 1, 1, 0, 0])
Dict{Int64,Int64} with 2 entries:
    0 => 3
    1 => 2
```
"""
function labelfreq end

"""
    labelfreq!(dict, obj) -> Dict

updates the given label-frequency-map `dict` with the absolute
frequencies for each label in `obj`

```juliarepl
julia> ld = labelfreq([0, 1, 1, 0, 0])
Dict{Int64,Int64} with 2 entries:
    0 => 3
    1 => 2

julia> labelfreq!(ld, [1,0,0])
Dict{Int64,Int64} with 2 entries:
    0 => 5
    1 => 3
```
"""
function labelfreq! end

"""
    labelmap2vec(dict) -> Vector

Inverse function of labelmap.
Computes an `array` of labels by element-wise
traversal of the entries in `dict`.

```juliarepl
julia> labelvec = [:yes,:no,:no,:yes,:yes]

julia> lm = labelmap(labelvec)
Dict{Symbol,Array{Int64,1}} with 2 entries:
    :yes => [1, 4, 5]
    :no  => [2, 3]

julia> labelmap2vec(lm)
5-element Array{Symbol,1}:
    :yes
    :no
    :no
    :yes
    :yes
```
"""
function labelmap2vec end
