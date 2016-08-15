using LearnBase
using Base.Test

# Test if types are exported properly
@test Cost <: Any
@test Penalty <: Cost
@test Loss <: Cost
@test UnsupervisedLoss <: Loss
@test SupervisedLoss <: Loss
@test MarginLoss <: SupervisedLoss
@test DistanceLoss <: SupervisedLoss

# Test if functions are exported properly
@test typeof(transform) <: Function
@test typeof(transform!) <: Function
@test typeof(getobs) <: Function
@test typeof(getobs!) <: Function
@test typeof(learn) <: Function
@test typeof(learn!) <: Function
@test typeof(update) <: Function
@test typeof(update!) <: Function

@test typeof(value) <: Function
@test typeof(value!) <: Function
@test typeof(meanvalue) <: Function
@test typeof(sumvalue) <: Function
@test typeof(meanderiv) <: Function
@test typeof(sumderiv) <: Function
@test typeof(deriv) <: Function
@test typeof(deriv2) <: Function
@test typeof(deriv!) <: Function
@test typeof(value_deriv) <: Function
@test typeof(grad) <: Function
@test typeof(grad!) <: Function
@test typeof(addgrad!) <: Function
@test typeof(value_grad) <: Function
@test typeof(value_grad!) <: Function
@test typeof(prox) <: Function
@test typeof(prox!) <: Function

@test typeof(isminimizable) <: Function
@test typeof(isdifferentiable) <: Function
@test typeof(istwicedifferentiable) <: Function
@test typeof(isconvex) <: Function
@test typeof(isstronglyconvex) <: Function
@test typeof(isnemitski) <: Function
@test typeof(isunivfishercons) <: Function
@test typeof(isfishercons) <: Function
@test typeof(islipschitzcont) <: Function
@test typeof(islocallylipschitzcont) <: Function
@test typeof(islipschitzcont_deriv) <: Function
@test typeof(isclipable) <: Function
@test typeof(ismarginbased) <: Function
@test typeof(isclasscalibrated) <: Function
@test typeof(isdistancebased) <: Function

@test typeof(issymmetric) <: Function

@test typeof(fit) <: Function
@test typeof(fit!) <: Function
@test typeof(nobs) <: Function

# Test that LearnBase reuses StatsBase functions
using StatsBase
@test StatsBase.fit  == LearnBase.fit
@test StatsBase.fit! == LearnBase.fit!
@test StatsBase.nobs == LearnBase.nobs

