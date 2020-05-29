# dummy types for testing
struct MyStronglyConvexType <: LearnBase.SupervisedLoss end
LearnBase.isstronglyconvex(::MyStronglyConvexType) = true
LearnBase.islipschitzcont(::MyStronglyConvexType) = true

@testset "Costs" begin
    @test LearnBase.Cost <: Any
    @test LearnBase.Penalty <: LearnBase.Cost
    @test LearnBase.Loss <: LearnBase.Cost
    @test LearnBase.UnsupervisedLoss <: LearnBase.Loss
    @test LearnBase.SupervisedLoss <: LearnBase.Loss
    @test LearnBase.MarginLoss <: LearnBase.SupervisedLoss
    @test LearnBase.DistanceLoss <: LearnBase.SupervisedLoss

    @test typeof(LearnBase.value) <: Function
    @test typeof(LearnBase.deriv) <: Function
    @test typeof(LearnBase.deriv2) <: Function

    @test typeof(LearnBase.isminimizable) <: Function
    @test typeof(LearnBase.isdifferentiable) <: Function
    @test typeof(LearnBase.istwicedifferentiable) <: Function
    @test typeof(LearnBase.isconvex) <: Function
    @test typeof(LearnBase.isstrictlyconvex) <: Function
    @test typeof(LearnBase.isstronglyconvex) <: Function
    @test typeof(LearnBase.isnemitski) <: Function
    @test typeof(LearnBase.islipschitzcont) <: Function
    @test typeof(LearnBase.islocallylipschitzcont) <: Function
    @test typeof(LearnBase.isfishercons) <: Function
    @test typeof(LearnBase.isunivfishercons) <: Function
    @test typeof(LearnBase.isclipable) <: Function
    @test typeof(LearnBase.ismarginbased) <: Function
    @test typeof(LearnBase.isdistancebased) <: Function
    @test typeof(LearnBase.isclasscalibrated) <: Function
    @test typeof(LearnBase.issymmetric) <: Function

    # test fallback methods
    @test LearnBase.isstronglyconvex(MyStronglyConvexType())
    @test LearnBase.isstrictlyconvex(MyStronglyConvexType())
    @test LearnBase.isconvex(MyStronglyConvexType())
    @test LearnBase.islipschitzcont(MyStronglyConvexType())
    @test LearnBase.islocallylipschitzcont(MyStronglyConvexType())
end
