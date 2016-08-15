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

abstract Cost

abstract Loss <: Cost
abstract SupervisedLoss <: Loss
abstract MarginLoss <: SupervisedLoss
abstract DistanceLoss <: SupervisedLoss
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
function isstronglyconvex end
function isnemitski end
function isunivfishercons end
function isfishercons end
function islipschitzcont end
function islocallylipschitzcont end
function islipschitzcont_deriv end
function isclipable end
function ismarginbased end
function isclasscalibrated end
function isdistancebased end

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

"""
An algorithm capable of mutating a `Minimizable` in
such a way that its `value` is minimal
"""
abstract Optimizer

function update end
function update! end
function learn end
function learn! end

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

    Optimizer,

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

    isminimizable,
    isdifferentiable,
    istwicedifferentiable,
    isconvex,
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
