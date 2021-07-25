@test typeof(LearnBase.getobs) <: Function
@test typeof(LearnBase.getobs!) <: Function
@test typeof(LearnBase.gettargets) <: Function
@test typeof(LearnBase.datasubset) <: Function

@testset "getobs" begin
    
    function LearnBase.getobs(x::AbstractArray{T,N}, idx; obsdim=default_obsdim(x)) where {T,N}   
        _idx = ntuple(i->  i == obsdim ? idx : Colon(), N)
        return x[_idx...]
    end
    StatsBase.nobs(x::AbstractArray; obsdim=default_obsdim(x)) = size(x, obsdim)  

    a = rand(2,3)
    @test nobs(a) == 3
    @test getobs(a, 1) ≈ a[:,1]
    @test getobs(a, 2) ≈ a[:,2]
    @test getobs(a, 1, obsdim=1) ≈ a[1,:]
    @test getobs(a, 2, obsdim=1) ≈ a[2,:]

    # Here we use Ref to protect idx against broadcasting
    LearnBase.getobs(t::Tuple, idx) = getobs.(t, Ref(idx))
    # Assume all elements have the same nummber of observations.
    # It would be safer to check explicitely though.
    StatsBase.nobs(t::Tuple) = nobs(t[1])

    # A dataset with 3 observations, each with 2 input features
    X, Y = rand(2, 3), rand(3)
    dataset = (X, Y) 

    o = getobs(dataset, 2) # -> (X[:,2], Y[2])
    @test o[1] ≈ X[:,2]
    @test o[2] == Y[2]

    o = getobs(dataset, 1:2) # -> (X[:,1:2], Y[1:2])
    @test o[1] ≈ X[:,1:2]
    @test o[2] == Y[1:2]
end


