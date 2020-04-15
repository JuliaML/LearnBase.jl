@testset "AggMode" begin
    @test typeof(LearnBase.AggMode.None()) <: LearnBase.AggregateMode
    @test typeof(LearnBase.AggMode.Sum()) <: LearnBase.AggregateMode
    @test typeof(LearnBase.AggMode.Mean()) <: LearnBase.AggregateMode
    @test typeof(LearnBase.AggMode.WeightedSum([1,2,3])) <: LearnBase.AggregateMode
    @test typeof(LearnBase.AggMode.WeightedMean([1,2,3])) <: LearnBase.AggregateMode
end
