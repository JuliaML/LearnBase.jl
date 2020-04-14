@doc doc"""
Baseclass for any kind of cost. Notable examples for
costs are `Loss` and `Penalty`.
"""
abstract type Cost end

@doc doc"""
Baseclass for all losses. A loss is some (possibly simplified)
function `L(features, targets, outputs)`, where `outputs` are the
result of some function `f(features)`.
"""
abstract type Loss <: Cost end

@doc doc"""
A loss is considered **supervised**, if all the information needed
to compute `L(features, targets, outputs)` are contained in
`targets` and `outputs`, and thus allows for the simplification
`L(targets, outputs)`.
"""
abstract type SupervisedLoss <: Loss end

@doc doc"""
A supervised loss is considered **unary** if it can be written as a composition
`L(Ψ(output, target))` for some binary function `Ψ`. In this case the loss can
be evaluated with a single argument termed the **agreement** `Ψ(output, target)`.
Notable examples for unary supervised losses are distance-based (`Ψ(ŷ,y) = ŷ - y`)
and margin-based (`Ψ(ŷ,y) = ŷ*y`) losses.
"""
abstract type UnarySupervisedLoss <: SupervisedLoss end

@doc doc"""
A supervised loss that can be simplified to
`L(targets, outputs) = L(targets - outputs)` is considered
**distance-based**.
"""
abstract type DistanceLoss <: UnarySupervisedLoss end

@doc doc"""
A supervised loss, where the targets are in {-1, 1}, and which
can be simplified to `L(targets, outputs) = L(targets * outputs)`
is considered **margin-based**.
"""
abstract type MarginLoss <: UnarySupervisedLoss end

@doc doc"""
A loss is considered **unsupervised**, if all the information needed
to compute `L(features, targets, outputs)` are contained in
`features` and `outputs`, and thus allows for the simplification
`L(features, outputs)`.
"""
abstract type UnsupervisedLoss <: Loss end

@doc doc"""
Baseclass for all penalties.
"""
abstract type Penalty <: Cost end

function value end
function value! end

function deriv end
function deriv! end

function deriv2 end
function deriv2! end

function value_deriv end
function value_deriv! end

@doc doc"""
    isconvex(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a convex function.
A function ``f : \mathbb{R}^n \rightarrow \mathbb{R}`` is convex if
its domain is a convex set and if for all ``x, y`` in that
domain, with ``\theta`` such that for ``0 \leq \theta \leq 1``,
we have

```math
f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)
```
"""
isconvex(l) = isstrictlyconvex(l)

@doc doc"""
    isstrictlyconvex(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a strictly convex function.
A function ``f : \mathbb{R}^n \rightarrow \mathbb{R}`` is
strictly convex if its domain is a convex set and if for all
``x, y`` in that domain where ``x \neq y``, with
``\theta`` such that for ``0 < \theta < 1``, we have

```math
f(\theta x + (1 - \theta) y) < \theta f(x) + (1 - \theta) f(y)
```
"""
isstrictlyconvex(l) = isstronglyconvex(l)

@doc doc"""
    isstronglyconvex(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a strongly convex function.
A function ``f : \mathbb{R}^n \rightarrow \mathbb{R}`` is
``m``-strongly convex if its domain is a convex set and if
``\forall x,y \in`` **dom** ``f`` where ``x \neq y``,
and ``\theta`` such that for ``0 \le \theta \le 1`` , we have

```math
f(\theta x + (1 - \theta)y) < \theta f(x) + (1 - \theta) f(y) - 0.5 m \cdot \theta (1 - \theta) {\| x - y \|}_2^2
```

In a more familiar setting, if the loss function is
differentiable we have

```math
\left( \nabla f(x) - \nabla f(y) \right)^\top (x - y) \ge m {\| x - y\|}_2^2
```
"""
isstronglyconvex(::SupervisedLoss) = false

@doc doc"""
    isdifferentiable(loss::SupervisedLoss, [x::Number]) -> Bool

Return `true` if the given `loss` is differentiable
(optionally limited to the given point `x` if specified).

A function ``f : \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}`` is
differentiable at a point ``x \in`` **int dom** ``f``, if there
exists a matrix ``Df(x) \in \mathbb{R}^{m \times n}`` such that
it satisfies:

```math
\lim_{z \neq x, z \to x} \frac{{\|f(z) - f(x) - Df(x)(z-x)\|}_2}{{\|z - x\|}_2} = 0
```

A function is differentiable if its domain is open and it is
differentiable at every point ``x``.
"""
isdifferentiable(l::SupervisedLoss) = istwicedifferentiable(l)
isdifferentiable(l::SupervisedLoss, at) = isdifferentiable(l)

@doc doc"""
    istwicedifferentiable(loss::SupervisedLoss, [x::Number]) -> Bool

Return `true` if the given `loss` is differentiable
(optionally limited to the given point `x` if specified).

A function ``f : \mathbb{R}^{n} \rightarrow \mathbb{R}`` is
said to be twice differentiable at a point ``x \in`` **int
dom** ``f``, if the function derivative for ``\nabla f``
exists at ``x``.

```math
\nabla^2 f(x) = D \nabla f(x)
```

A function is twice differentiable if its domain is open and it
is twice differentiable at every point ``x``.
"""
istwicedifferentiable(::SupervisedLoss) = false
istwicedifferentiable(l::SupervisedLoss, at) = istwicedifferentiable(l)

@doc doc"""
    islocallylipschitzcont(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` function is locally-Lipschitz
continous.

A supervised loss ``L : Y \times \mathbb{R} \rightarrow [0, \infty)``
is called locally Lipschitz continuous if ``\forall a \ge 0``
there exists a constant :math:`c_a \ge 0`, such that

```math
\sup_{y \in Y} \left| L(y,t) − L(y,t′) \right| \le c_a |t − t′|,  \qquad  t,t′ \in [−a,a]
```

Every convex function is locally lipschitz continuous.
"""
islocallylipschitzcont(l) = isconvex(l) || islipschitzcont(l)

@doc doc"""
    islipschitzcont(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` function is Lipschitz continuous.

A supervised loss function ``L : Y \times \mathbb{R} \rightarrow
[0, \infty)`` is Lipschitz continous, if there exists a finite
constant ``M < \infty`` such that

```math
|L(y, t) - L(y, t′)| \le M |t - t′|,  \qquad  \forall (y, t) \in Y \times \mathbb{R}
```
"""
islipschitzcont(::SupervisedLoss) = false

@doc doc"""
    isnemitski(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a Nemitski loss function.

We call a supervised loss function ``L : Y \times \mathbb{R}
\rightarrow [0,\infty)`` a Nemitski loss if there exist a
measurable function ``b : Y \rightarrow [0, \infty)`` and an
increasing function ``h : [0, \infty) \rightarrow [0, \infty)``
such that

```math
L(y,\hat{y}) \le b(y) + h(|\hat{y}|),  \qquad  (y, \hat{y}) \in Y \times \mathbb{R}.
```

If a loss if locally lipsschitz continuous then it is a Nemitski loss
"""
isnemitski(l::SupervisedLoss) = islocallylipschitzcont(l)

@doc doc"""
    isclipable(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` function is clipable. A
supervised loss ``L : Y \times \mathbb{R} \rightarrow [0,
\infty)`` can be clipped at ``M > 0`` if, for all ``(y,t)
\in Y \times \mathbb{R}``,

```math
L(y, \hat{t}) \le L(y, t)
```

where ``\hat{t}`` denotes the clipped value of ``t`` at
``\pm M``. That is

```math
\hat{t} = \begin{cases} -M & \quad \text{if } t < -M \\ t & \quad \text{if } t \in [-M, M] \\ M & \quad \text{if } t > M \end{cases}
```
"""
isclipable(::SupervisedLoss) = false

@doc doc"""
    isdistancebased(loss::SupervisedLoss) -> Bool

Return `true` ifthe given `loss` is a distance-based loss.

A supervised loss function ``L : Y \times \mathbb{R} \rightarrow
[0, \infty)`` is said to be **distance-based**, if there exists a
representing function ``\psi : \mathbb{R} \rightarrow [0, \infty)``
satisfying ``\psi (0) = 0`` and

```math
L(y, \hat{y}) = \psi (\hat{y} - y),  \qquad  (y, \hat{y}) \in Y \times \mathbb{R}
```
"""
isdistancebased(::SupervisedLoss) = false

@doc doc"""
    ismarginbased(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` is a margin-based loss.

A supervised loss function ``L : Y \times \mathbb{R} \rightarrow
[0, \infty)`` is said to be **margin-based**, if there exists a
representing function ``\psi : \mathbb{R} \rightarrow [0, \infty)``
satisfying

```math
L(y, \hat{y}) = \psi (y \cdot \hat{y}),  \qquad  (y, \hat{y}) \in Y \times \mathbb{R}
```
"""
ismarginbased(::SupervisedLoss) = false

@doc doc"""
    isclasscalibrated(loss::SupervisedLoss) -> Bool
"""
isclasscalibrated(::SupervisedLoss) = false

@doc doc"""
    issymmetric(loss::SupervisedLoss) -> Bool

Return `true` if the given loss is a symmetric loss.

A function ``f : \mathbb{R} \rightarrow [0,\infty)`` is said
to be symmetric about origin if we have

```math
f(x) = f(-x), \qquad  \forall x \in \mathbb{R}
```

A distance-based loss is said to be symmetric if its representing
function is symmetric.
"""
issymmetric(::SupervisedLoss) = false

@doc doc"""
    isminimizable(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` is a minimizable loss.
"""
isminimizable(l) = isconvex(l)
