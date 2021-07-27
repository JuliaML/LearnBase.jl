using LearnBase
using StatsBase
using SparseArrays
using Test


@testset "Observation" begin 
    include("observation.jl")
end

@testset "Observation" begin 
    include("iteration.jl")
end

include("aggmode.jl")
include("costs.jl")
include("other.jl")
