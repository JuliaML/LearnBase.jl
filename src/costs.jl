"""
Baseclass for any kind of cost. Notable examples for
costs are `Loss` and `Penalty`.
"""
abstract type Cost end

"""
Baseclass for all losses. A loss is some (possibly simplified)
function `L(x, y, ŷ)`, of features `x`, targets `y` and outputs
`ŷ = f(x)` for some function `f`.
"""
abstract type Loss <: Cost end

"""
A loss is considered **supervised**, if all the information needed
to compute `L(x, y, ŷ)` are contained in `y` and `ŷ`, and thus allows
for the simplification `L(y, ŷ)`.
"""
abstract type SupervisedLoss <: Loss end

"""
A supervised loss is considered **unary** if it can be written as a composition
`L(β(y, ŷ))` for some binary function `β`. In this case the loss can be evaluated
with a single argument termed the **agreement** `β(y, ŷ)`. Notable
examples for unary supervised losses are distance-based (`L(y,ŷ) = ψ(ŷ - y)`)
and margin-based (`L(y,ŷ) = ψ(y⋅ŷ)`) losses.
"""
abstract type UnarySupervisedLoss <: SupervisedLoss end

"""
A supervised loss that can be simplified to `L(y, ŷ) = L(ŷ - y)`
is considered **distance-based**.
"""
abstract type DistanceLoss <: UnarySupervisedLoss end

"""
A supervised loss with targets `y ∈ {-1, 1}`, and which
can be simplified to `L(y, ŷ) = L(y⋅ŷ)` is considered
**margin-based**.
"""
abstract type MarginLoss <: UnarySupervisedLoss end

"""
A loss is considered **unsupervised**, if all the information needed
to compute `L(x, y, ŷ)` are contained in `x` and `ŷ`, and thus allows
for the simplification `L(x, ŷ)`.
"""
abstract type UnsupervisedLoss <: Loss end

"""
Baseclass for all penalties.
"""
abstract type Penalty <: Cost end

"""
    value(loss, target, output)

Compute the (non-negative) numeric result for the `loss` function
return it. Note that `target` and `output` can be of different
numeric type, in which case promotion is performed in the manner
appropriate for the given loss `L: Y × ℝ → [0,∞)`.

# Arguments

- `loss::SupervisedLoss`: The loss function `L` we want to compute.
- `target::Number`: The ground truth `y ∈ Y` of the observation.
- `output::Number`: The predicted output `ŷ ∈ ℝ`.
  for the observation.
"""
value(loss::SupervisedLoss, target::Number, output::Number) =
    MethodError(value, (loss, target, output))

"""
    value(loss, targets, outputs, aggmode) -> Number

Compute the weighted or unweighted sum or mean (depending on
aggregation mode `aggmode`) of the individual values of the `loss`
function for each pair in `targets` and `outputs`. This method
will not allocate a temporary array.

In the case that the two parameters are arrays with a different
number of dimensions, broadcast will be performed. Note that the
given parameters are expected to have the same size in the
dimensions they share.

# Arguments

- `loss::SupervisedLoss`: The loss function `L` we want to compute.
- `targets::AbstractArray`: The array of ground truths `𝐲`.
- `outputs::AbstractArray`: The array of predicted outputs `𝐲̂`.
- `aggmode::AggregateMode`: Must be one of the following:
  [`AggMode.Sum()`](@ref), [`AggMode.Mean()`](@ref),
  [`AggMode.WeightedSum`](@ref), or [`AggMode.WeightedMean`](@ref).
"""
value(loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray, aggmode::AggregateMode) =
    MethodError(value, (loss, targets, outputs, aggmode))

"""
    value(loss, targets, outputs, aggmode, obsdim) -> AbstractVector

Compute the values of the `loss` function for each pair in
`targets` and `outputs` individually, and return either the
weighted or unweighted sum or mean for each observation (depending on
`aggmode`). This method will not allocate a temporary array, but
it will allocate the resulting vector.

Both arrays have to be of the same shape and size. Furthermore
they have to have at least two array dimensions (i.e. they must
not be vectors).

# Arguments

- `loss::SupervisedLoss`: The loss function `L` we want to compute.
- `targets::AbstractArray`: The array of ground truths `𝐲`.
- `outputs::AbstractArray`: The array of predicted outputs `𝐲̂`.
- `aggmode::AggregateMode`: Must be one of the following:
  [`AggMode.Sum()`](@ref), [`AggMode.Mean()`](@ref),
  [`AggMode.WeightedSum`](@ref), or [`AggMode.WeightedMean`](@ref).
- `obsdim::ObsDimension`: Specifies which of the array dimensions
  denotes the observations. See `?ObsDim` for more information.
"""
value(loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray,
      aggmode::AggregateMode, obsdim::ObsDimension) =
    MethodError(value, (loss, targets, outputs, aggmode, obsdim))

"""
    value!(buffer, loss, targets, outputs, aggmode, obsdim) -> buffer

Compute the values of the `loss` function for each pair in
`targets` and `outputs` individually, and return either the
weighted or unweighted sum or mean for each observation,
depending on `aggmode`. The results are stored into the given
vector `buffer`. This method will not allocate a temporary array.

Both arrays have to be of the same shape and size. Furthermore
they have to have at least two array dimensions (i.e. so they
must not be vectors).

# Arguments

- `buffer::AbstractArray`: Array to store the computed values in.
  Old values will be overwritten and lost.
- `loss::SupervisedLoss`: The loss function `L` we want to compute.
- `targets::AbstractArray`: The array of ground truths `𝐲`.
- `outputs::AbstractArray`: The array of predicted outputs `𝐲̂`.
- `aggmode::AggregateMode`: Must be one of the following:
  [`AggMode.Sum()`](@ref), [`AggMode.Mean()`](@ref),
  [`AggMode.WeightedSum`](@ref), or [`AggMode.WeightedMean`](@ref).
- `obsdim::ObsDimension`: Specifies which of the array dimensions
  denotes the observations. See `?ObsDim` for more information.
"""
value!(buffer::AbstractArray, loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray,
       aggmode::AggregateMode, obsdim::ObsDimension) =
    MethodError(value!, (buffer, loss, targets, outputs, aggmode, obsdim))

"""
    deriv(loss, target, output) -> Number

Compute the derivative for the `loss` function
in respect to the `output`. Note that `target` and
`output` can be of different numeric type, in which
case promotion is performed in the manner appropriate
for the given loss.

# Arguments

- `loss::SupervisedLoss`: The loss function `L` we want to compute.
- `target::Number`: The ground truth `y ∈ Y` of the observation.
- `output::Number`: The predicted output `ŷ ∈ ℝ` for the observation.
"""
deriv(loss::SupervisedLoss, target::Number, output::Number) =
    MethodError(deriv, (loss, target, output))

"""
    deriv(loss, targets, outputs, aggmode) -> Number

Compute the weighted or unweighted sum or mean (depending on `aggmode`)
of the individual derivatives of the `loss` function for each pair
in `targets` and `outputs`. This method will not allocate a
temporary array.

In the case that the two parameters are arrays with a different
number of dimensions, broadcast will be performed. Note that the
given parameters are expected to have the same size in the
dimensions they share.

# Arguments

- `loss::SupervisedLoss`: The loss function `L` we want to compute.
- `targets::AbstractArray`: The array of ground truths `𝐲`.
- `outputs::AbstractArray`: The array of predicted outputs `𝐲̂`.
- `aggmode::AggregateMode`: Must be one of the following:
  [`AggMode.Sum()`](@ref), [`AggMode.Mean()`](@ref),
  [`AggMode.WeightedSum`](@ref), or [`AggMode.WeightedMean`](@ref).
"""
deriv(loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray, aggmode::AggregateMode) =
    MethodError(deriv, (loss, targets, outputs, aggmode))

"""
    deriv(loss, targets, outputs, aggmode, obsdim) -> AbstractVector

Compute the derivative of the `loss` function for each pair in
`targets` and `outputs` individually, and return either the
weighted or unweighted sum or mean for each observation (depending on
`aggmode`). This method will not allocate a temporary array, but
it will allocate the resulting vector.

Both arrays have to be of the same shape and size. Furthermore
they have to have at least two array dimensions (i.e. they must
not be vectors).

# Arguments

- `loss::SupervisedLoss`: The loss function `L` we want to compute.
- `targets::AbstractArray`: The array of ground truths `𝐲`.
- `outputs::AbstractArray`: The array of predicted outputs `𝐲̂`.
- `aggmode::AggregateMode`: Must be one of the following:
  [`AggMode.Sum()`](@ref), [`AggMode.Mean()`](@ref),
  [`AggMode.WeightedSum`](@ref), or [`AggMode.WeightedMean`](@ref).
- `obsdim::ObsDimension`: Specifies which of the array dimensions
  denotes the observations. See `?ObsDim` for more information.
"""
deriv(loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray,
      aggmode::AggregateMode, obsdim::ObsDimension) =
    MethodError(deriv, (loss, targets, outputs, aggmode, obsdim))

"""
    deriv!(buffer, loss, targets, outputs, aggmode, obsdim) -> buffer

Compute the derivative of the `loss` function for each pair in
`targets` and `outputs` individually, and return either the
weighted or unweighted sum or mean for each observation, depending on
`aggmode`. The results are stored into the given vector `buffer`.
This method will not allocate a temporary array.

Both arrays have to be of the same shape and size. Furthermore
they have to have at least two array dimensions (i.e. so they
must not be vectors).

# Arguments

- `buffer::AbstractArray`: Array to store the computed values in.
- `loss::SupervisedLoss`: The loss function `L` we want to compute.
- `targets::AbstractArray`: The array of ground truths `𝐲`.
- `outputs::AbstractArray`: The array of predicted outputs `𝐲̂`.
- `aggmode::AggregateMode`: Must be one of the following:
  [`AggMode.Sum()`](@ref), [`AggMode.Mean()`](@ref),
  [`AggMode.WeightedSum`](@ref), or [`AggMode.WeightedMean`](@ref).
- `obsdim::ObsDimension`: Specifies which of the array dimensions
  denotes the observations. See `?ObsDim` for more information.
"""
deriv!(buffer::AbstractArray, loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray,
       aggmode::AggregateMode, obsdim::ObsDimension) =
    MethodError(deriv!, (buffer, loss, targets, outputs, aggmode, obsdim))

"""
    deriv2(loss, target, output) -> Number

Compute the second derivative for the `loss` function
in respect to the `output`. Note that `target` and `output`
can be of different numeric type, in which case promotion is
performed in the manner appropriate for the given loss.

# Arguments

- `loss::SupervisedLoss`: The loss function `L` we want to compute.
- `target::Number`: The ground truth `y ∈ Y` of the observation.
- `output::Number`: The predicted output `ŷ ∈ ℝ` for the observation.
"""
deriv2(loss::SupervisedLoss, target::Number, output::Number) =
    MethodError(deriv2, (loss, target, output))

"""
    deriv2(loss, targets, outputs, aggmode) -> Number

Compute the weighted or unweighted sum or mean (depending on `aggmode`)
of the individual second derivatives of the `loss` function for
each pair in `targets` and `outputs`. This method will not
allocate a temporary array.

In the case that the two parameters are arrays with a different
number of dimensions, broadcast will be performed. Note that the
given parameters are expected to have the same size in the
dimensions they share.

# Arguments

- `loss::SupervisedLoss`: The loss function `L` we want to compute.
- `targets::AbstractArray`: The array of ground truths `𝐲`.
- `outputs::AbstractArray`: The array of predicted outputs `𝐲̂`.
- `aggmode::AggregateMode`: Must be one of the following:
  [`AggMode.Sum()`](@ref), [`AggMode.Mean()`](@ref),
  [`AggMode.WeightedSum`](@ref), or [`AggMode.WeightedMean`](@ref).
"""
deriv2(loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray, aggmode::AggregateMode) =
    MethodError(deriv2, (loss, targets, outputs, aggmode))

"""
    deriv2(loss, targets, outputs, aggmode, obsdim) -> AbstractVector

Compute the second derivative of the `loss` function for each pair in
`targets` and `outputs` individually, and return either the
weighted or unweighted sum or mean for each observation (depending on
`aggmode`). This method will not allocate a temporary array, but
it will allocate the resulting vector.

Both arrays have to be of the same shape and size. Furthermore
they have to have at least two array dimensions (i.e. they must
not be vectors).

# Arguments

- `loss::SupervisedLoss`: The loss function `L` we want to compute.
- `targets::AbstractArray`: The array of ground truths `𝐲`.
- `outputs::AbstractArray`: The array of predicted outputs `𝐲̂`.
- `aggmode::AggregateMode`: Must be one of the following:
  [`AggMode.Sum()`](@ref), [`AggMode.Mean()`](@ref),
  [`AggMode.WeightedSum`](@ref), or [`AggMode.WeightedMean`](@ref).
- `obsdim::ObsDimension`: Specifies which of the array dimensions
  denotes the observations. See `?ObsDim` for more information.
"""
deriv2(loss::SupervisedLoss, targets::AbstractArray, outputs::AbstractArray,
       aggmode::AggregateMode, obsdim::ObsDimension) =
    MethodError(deriv2, (loss, targets, outputs, aggmode, obsdim))

function deriv2! end
function value_deriv end
function value_deriv! end

"""
    isconvex(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a convex function.
A function `f: ℝⁿ → ℝ` is convex if its domain is a convex set
and if for all `x, y` in that domain, with `θ` such that for
`0 ≦ θ ≦ 1`, we have `f(θ x + (1 - θ) y) ≦ θ f(x) + (1 - θ) f(y)`.
"""
isconvex(l) = isstrictlyconvex(l)

"""
    isstrictlyconvex(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a strictly convex function.
A function `f : ℝⁿ → ℝ` is strictly convex if its domain is a convex
set and if for all `x, y` in that domain where `x ≠ y`, with `θ` such
that for `0 < θ < 1`, we have `f(θ x + (1 - θ) y) < θ f(x) + (1 - θ) f(y)`.
"""
isstrictlyconvex(l) = isstronglyconvex(l)

"""
    isstronglyconvex(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a strongly convex function.
A function `f : ℝⁿ → ℝ` is `m`-strongly convex if its domain is a convex
set, and if for all `x, y` in that domain where `x ≠ y`, and `θ` such that
for `0 ≤ θ ≤ 1`, we have
`f(θ x + (1 - θ)y) < θ f(x) + (1 - θ) f(y) - 0.5 m ⋅ θ (1 - θ) | x - y |₂²`

In a more familiar setting, if the loss function is differentiable we have
`(∇f(x) - ∇f(y))ᵀ (x - y) ≥ m | x - y |₂²`
"""
isstronglyconvex(::SupervisedLoss) = false

"""
    isdifferentiable(loss::SupervisedLoss, [x::Number]) -> Bool

Return `true` if the given `loss` is differentiable
(optionally limited to the given point `x` if specified).

A function `f : ℝⁿ → ℝᵐ` is differentiable at a point `x` in the interior
domain of `f` if there exists a matrix `Df(x) ∈ ℝ^(m × n)` such that
it satisfies:

`lim_{z ≠ x, z → x} (|f(z) - f(x) - Df(x)(z-x)|₂) / |z - x|₂ = 0`

A function is differentiable if its domain is open and it is
differentiable at every point `x`.
"""
isdifferentiable(l::SupervisedLoss) = istwicedifferentiable(l)
isdifferentiable(l::SupervisedLoss, at) = isdifferentiable(l)

"""
    istwicedifferentiable(loss::SupervisedLoss, [x::Number]) -> Bool

Return `true` if the given `loss` is differentiable
(optionally limited to the given point `x` if specified).

A function `f : ℝⁿ → ℝ` is said to be twice differentiable
at a point `x` in the interior domain of `f`, if the function
derivative for `∇f` exists at `x`: `∇²f(x) = D∇f(x)`.

A function is twice differentiable if its domain is open and it
is twice differentiable at every point `x`.
"""
istwicedifferentiable(::SupervisedLoss) = false
istwicedifferentiable(l::SupervisedLoss, at) = istwicedifferentiable(l)

"""
    islocallylipschitzcont(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` function is locally-Lipschitz
continous.

A supervised loss `L : Y × ℝ → [0, ∞)` is called locally Lipschitz
continuous if for all `a ≥ 0` there exists a constant `cₐ ≥ 0`,
such that

`sup_{y ∈ Y} | L(y,t) − L(y,t′) | ≤ cₐ |t − t′|, t, t′ ∈ [−a,a]`

Every convex function is locally lipschitz continuous.
"""
islocallylipschitzcont(l) = isconvex(l) || islipschitzcont(l)

"""
    islipschitzcont(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` function is Lipschitz continuous.

A supervised loss function `L : Y × ℝ → [0, ∞)` is Lipschitz continous,
if there exists a finite constant `M < ∞` such that
`|L(y, t) - L(y, t′)| ≤ M |t - t′|, ∀ (y, t) ∈ Y × ℝ`
"""
islipschitzcont(::SupervisedLoss) = false

"""
    isnemitski(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a Nemitski loss function.

We call a supervised loss function `L : Y × ℝ → [0,∞)` a Nemitski
loss if there exist a measurable function `b : Y → [0, ∞)` and an
increasing function `h : [0, ∞) → [0, ∞)` such that
`L(y,ŷ) ≤ b(y) + h(|ŷ|), (y, ŷ) ∈ Y × ℝ`

If a loss if locally lipsschitz continuous then it is a Nemitski loss.
"""
isnemitski(l::SupervisedLoss) = islocallylipschitzcont(l)

"""
    isclipable(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` function is clipable. A
supervised loss `L : Y × ℝ → [0,∞)` can be clipped at `M > 0`
if, for all `(y,t) ∈ Y × ℝ`, `L(y, t̂) ≤ L(y, t)` where
`t̂` denotes the clipped value of `t` at `± M`.
That is
`t̂ = -M` if `t < -M`, `t̂ = t` if `t ∈ [-M, M]`, and `t = M` if `t > M`.
"""
isclipable(::SupervisedLoss) = false

"""
    isdistancebased(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` is a distance-based loss.

A supervised loss function `L : Y × ℝ → [0,∞)` is said to be
distance-based, if there exists a representing function `ψ : ℝ → [0,∞)`
satisfying `ψ(0) = 0` and `L(y, ŷ) = ψ (ŷ - y), (y, ŷ) ∈ Y × ℝ`.
"""
isdistancebased(::SupervisedLoss) = false

"""
    ismarginbased(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` is a margin-based loss.

A supervised loss function `L : Y × ℝ → [0,∞)` is said to be
margin-based, if there exists a representing function `ψ : ℝ → [0,∞)`
satisfying `L(y, ŷ) = ψ(y⋅ŷ), (y, ŷ) ∈ Y × ℝ`.
"""
ismarginbased(::SupervisedLoss) = false

"""
    isclasscalibrated(loss::SupervisedLoss) -> Bool
"""
isclasscalibrated(::SupervisedLoss) = false

"""
    issymmetric(loss::SupervisedLoss) -> Bool

Return `true` if the given loss is a symmetric loss.

A function `f : ℝ → [0,∞)` is said to be symmetric
about origin if we have `f(x) = f(-x), ∀ x ∈ ℝ`.

A distance-based loss is said to be symmetric if its
representing function is symmetric.
"""
issymmetric(::SupervisedLoss) = false

"""
    isminimizable(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` is a minimizable loss.
"""
isminimizable(l) = isconvex(l)
