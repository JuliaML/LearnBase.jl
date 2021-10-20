using LearnBase: getobs, nobs, default_obsdim

@test typeof(LearnBase.getobs) <: Function
@test typeof(LearnBase.getobs!) <: Function
@test typeof(LearnBase.gettargets) <: Function
@test typeof(LearnBase.datasubset) <: Function

@testset "getobs and nobs" begin

    @testset "array" begin
        a = rand(2,3)
        @test nobs(a) == 3
        @test @inferred getobs(a, 1) == a[:,1]
        @test @inferred getobs(a, 2) == a[:,2]
        @test @inferred getobs(a, 1:2) == a[:,1:2]
        @test @inferred getobs(a, 1, 1) == a[1,:]
        @test @inferred getobs(a, 2, 1) == a[2,:]
        @test @inferred getobs(a, 2, nothing) ≈ a[:,2]
    end

    @testset "tuple" begin
        # A dataset with 3 observations, each with 2 input features
        X, Y = rand(2, 3), rand(3)
        dataset = (X, Y) 
        @test nobs(dataset) == 3
        if VERSION >= v"1.6"
            o = @inferred getobs(dataset, 2)
        else
            o = getobs(dataset, 2)
        end
        @test o[1] == X[:,2]
        @test o[2] == Y[2]

        if VERSION >= v"1.6"
            o = @inferred getobs(dataset, 1:2)
        else
            o = getobs(dataset, 1:2)
        end

        @test o[1] == X[:,1:2]
        @test o[2] == Y[1:2]
    end


    @testset "named tuple" begin
        X, Y = rand(2, 3), rand(3)
        dataset = (x=X, y=Y) 
        @test nobs(dataset) == 3
        if VERSION >= v"1.6"
            o = @inferred getobs(dataset, 2)
        else
            o = getobs(dataset, 2)
        end
        @test o.x == X[:,2]
        @test o.y == Y[2]

        if VERSION >= v"1.6"
            o = @inferred getobs(dataset, 1:2)
        else
            o = getobs(dataset, 1:2)
        end
        @test o.x == X[:,1:2]
        @test o.y == Y[1:2]
    end

    @testset "dict" begin
        X, Y = rand(2, 3), rand(3)
        dataset = Dict("X" => X, "Y" => Y) 
        @test nobs(dataset) == 3

        # o = @inferred getobs(dataset, 2) # not inferred
        o = getobs(dataset, 2)
        @test o["X"] == X[:,2]
        @test o["Y"] == Y[2]

        o = getobs(dataset, 1:2)
        @test o["X"] == X[:,1:2]
        @test o["Y"] == Y[1:2]
    end
end
