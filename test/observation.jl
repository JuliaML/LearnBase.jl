using LearnBase: getobs, nobs, default_obsdim, getobs!

@test typeof(LearnBase.getobs) <: Function
@test typeof(LearnBase.getobs!) <: Function
@test typeof(LearnBase.gettargets) <: Function

@testset "getobs and nobs" begin

    @testset "array" begin
        a = rand(2,3)
        @test nobs(a) == 3
        @test @inferred getobs(a, 1) == a[:,1]
        @test @inferred getobs(a, 2) == a[:,2]
        @test @inferred getobs(a, 1:2) == a[:,1:2]
        @test @inferred getobs(a, 1; obsdim = 1) == a[1,:]
        @test @inferred getobs(a, 2; obsdim = 1) == a[2,:]
        @test @inferred getobs(a, 2; obsdim = nothing) ≈ a[:,2]
    end

    @testset "0-dim SubArray" begin
        v = view([3], 1)
        @test @inferred(nobs(v)) === 1
        @test @inferred(getobs(v, 1)) === 3
        @test_throws BoundsError getobs(v, 2)
        @test_throws BoundsError getobs(v, 2:3)
    end

    @testset "getobs!" begin
        @testset "Array and Subarray" begin
            X = rand(4, 150)
            Xbuf = similar(X)
            y = repeat(["setosa","versicolor","virginica"], inner = 50)
            Y = permutedims(hcat(y,y), [2,1])
            Yt = hcat(y,y)
            yt = Y[1:1,:]
            Xv = view(X,:,:)
            yv = view(y,:)
            XX = rand(20,30,150)
            # interpreted as idx
            @test_throws Exception getobs!(Xbuf, X; obsdim = 1)
            # obsdim not defined without some idx
            @test_throws MethodError getobs!(Xbuf, X)
            @test_throws MethodError getobs!(Xbuf, X; obsdim = 1)
            # access outside nobs bounds
            @test_throws BoundsError getobs!(Xbuf, X, -1)
            @test_throws BoundsError getobs!(Xbuf, X, 0)
            @test_throws BoundsError getobs!(Xbuf, X, 0; obsdim = 1)
            @test_throws BoundsError getobs!(Xbuf, X, 151)
            @test_throws BoundsError getobs!(Xbuf, X, 151; obsdim = 2)
            @test_throws BoundsError getobs!(Xbuf, X, 151; obsdim = 1)
            @test_throws BoundsError getobs!(Xbuf, X, 5; obsdim = 1)
            xbuf1 = zeros(4)
            xbuf2 = zeros(4)
            @test @inferred(getobs!(xbuf1, X, 45)) == getobs!(xbuf2, X', 45, obsdim = 1)
            Xbuf1 = zeros(4,8)
            Xbuf2 = zeros(8,4)
            @test @inferred(getobs!(Xbuf1, X, 3:10)) == getobs!(Xbuf2, X', 3:10, obsdim = 1)'
            # obsdim = 2
            Xbuf1 = zeros(20,150)
            @test @inferred(getobs!(Xbuf1, XX, 5; obsdim = 2)) == XX[:,5,:]
            @test getobs!(Xbuf1, XX, 11, obsdim = 2) == XX[:,11,:]
            Xbuf2 = zeros(20,5,150)
            @test @inferred(getobs!(Xbuf2, XX, 6:10; obsdim = 2)) == XX[:,6:10,:]
            @test getobs!(Xbuf2, XX, 11:15, obsdim = 2) == XX[:,11:15,:]
        end
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

    # @testset "dict" begin
    #     X, Y = rand(2, 3), rand(3)
    #     dataset = Dict("X" => X, "Y" => Y) 
    #     @test nobs(dataset) == 3

    #     # o = @inferred getobs(dataset, 2) # not inferred
    #     o = getobs(dataset, 2)
    #     @test o["X"] == X[:,2]
    #     @test o["Y"] == Y[2]

    #     o = getobs(dataset, 1:2)
    #     @test o["X"] == X[:,1:2]
    #     @test o["Y"] == Y[1:2]
    # end
end
