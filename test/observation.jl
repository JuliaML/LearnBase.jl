using LearnBase: getobs, nobs, default_obsdim, getobs!
using SparseArrays

@test typeof(LearnBase.getobs) <: Function
@test typeof(LearnBase.getobs!) <: Function
@test typeof(LearnBase.gettargets) <: Function

X = rand(4, 150)
y = repeat(["setosa","versicolor","virginica"], inner = 50)
Y = permutedims(hcat(y,y), [2,1])
Xs = sprand(10,150,.5)
ys = sprand(150,.5)
Yt = hcat(y,y)
yt = Y[1:1,:]
Xv = view(X,:,:)
yv = view(y,:)
XX = rand(20,30,150)
XXX = rand(3,20,30,150)
X1 = hcat((1:150 for i = 1:10)...)'
Y1 = collect(1:150)
vars = (X, Xv, yv, XX, XXX, y)

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

    @testset "nobs" begin
        @test_throws MethodError nobs(X,X)
        @test_throws MethodError nobs(X,y)
    
        @testset "Array, SparseArray, and Tuple" begin
            @test_throws DimensionMismatch nobs((X,XX,rand(100)))
            @test_throws DimensionMismatch nobs((X,X'))
            @test_throws DimensionMismatch nobs((X,XX); obsdim = 1)
            for var in (Xs, ys, vars...)
                @test @inferred(nobs(var, obsdim = ndims(var))) === 150
                @test @inferred(nobs(var, obsdim = 100)) === 1
                @test @inferred(nobs(var)) === 150
            end
            @test @inferred(nobs(())) === 0
            @test @inferred(nobs((), obsdim = 1)) === 0
            @test @inferred(nobs((), obsdim = 3)) === 0
        end
    
        @testset "SubArray" begin
            @test @inferred(nobs(view(X,:,:))) === 150
            @test @inferred(nobs(view(X,:,:))) === 150
            @test @inferred(nobs(view(XX,:,:,:))) === 150
            @test @inferred(nobs(view(XXX,:,:,:,:))) === 150
            @test @inferred(nobs(view(y,:))) === 150
            @test @inferred(nobs(view(Y,:,:))) === 150
        end
    
        @testset "various obsdim" begin
            @test_throws TypeError nobs(X, obsdim = 1.0)
            @test_throws DimensionMismatch nobs((X',X); obsdim = (1, ndims(X'), ndims(X)))
            @test_throws DimensionMismatch nobs((X',X); obsdim = (1,))
            @test_throws DimensionMismatch nobs((X',X); obsdim = (1, 2, 2))
            @test @inferred(nobs(Xs; obsdim = 1)) === 10
            @test @inferred(nobs(XXX; obsdim = 1)) === 3
            @test @inferred(nobs(XXX; obsdim = 2)) === 20
            @test @inferred(nobs(XXX; obsdim = 3)) === 30
            @test @inferred(nobs(XXX; obsdim = 4)) === 150
            @test @inferred(nobs((X,y); obsdim = (ndims(X), ndims(y)))) === 150
            @test @inferred(nobs((X',y); obsdim = 1)) === 150
            @test @inferred(nobs((X',X'); obsdim = 1)) === 150
            @test @inferred(nobs((X',X); obsdim = (1, ndims(X)))) === 150
            @test @inferred(nobs((X',X); obsdim = (1, 2))) === 150
            @test @inferred(nobs((X',X,X); obsdim = (1, 2, 2))) === 150
            @test @inferred(nobs((X, X); obsdim = 1)) === 4
        end
    end

    @testset "getobs" begin
        @testset "Array and Subarray" begin
            # access outside nobs bounds
            @test_throws BoundsError getobs(X, -1)
            @test_throws BoundsError getobs(X, 0)
            @test_throws BoundsError getobs(X, 0; obsdim = 1)
            @test_throws BoundsError getobs(X, 151)
            @test_throws BoundsError getobs(X, 151; obsdim = 2)
            @test_throws BoundsError getobs(X, 151; obsdim = 1)
            @test_throws BoundsError getobs(X, 5; obsdim = 1)
            @test_throws MethodError getobs(X; obsdim = 1)
            @test getobs(X, 45) == getobs(X', 45; obsdim = 1)
            @test getobs(X, 3:10) == getobs(X', 3:10; obsdim = 1)'
            for i in (2, 2:20, [2,1,4])
                @test getobs(XX, i) == XX[:, :, i]
                @test getobs(XX, i; obsdim = 1) == XX[i, :, :]
                @test getobs(XX, i; obsdim = 2) == XX[:, i, :]
            end
            for i in (2, 1:150, 2:10, [2,5,7], [2,1])
                @test_throws BoundsError getobs(X, i; obsdim = 12)
                @test typeof(getobs(Xv, i)) <: Array
                @test typeof(getobs(yv, i)) <: ((i isa Int) ? String : Array)
                @test all(getobs(Xv, i) .== X[:, i])
                @test getobs(Xv,i)  == X[:,i]
                @test getobs(X,i)   == X[:,i]
                @test getobs(XX,i)  == XX[:,:,i]
                @test getobs(XXX,i) == XXX[:,:,:,i]
                @test getobs(y,i)   == y[i]
                @test getobs(yv,i)  == y[i]
                @test getobs(Y,i)   == Y[:,i]
            end
        end
    
        @testset "SparseArray" begin
            @test getobs(Xs, 45) == getobs(Xs', 45, obsdim = 1)
            @test getobs(Xs, 3:9) == getobs(Xs', 3:9, obsdim = 1)'
            @test typeof(getobs(Xs,2)) <: SparseVector
            @test typeof(getobs(Xs,1:5)) <: SparseMatrixCSC
            @test typeof(getobs(ys,2)) <: Float64
            @test typeof(getobs(ys,1:5)) <: SparseVector
            for i in (2, 2:10, [2,1,4])
                @test getobs(Xs, i, obsdim = 1) == Xs[i,:]
                @test getobs(Xs, i, obsdim = 2) == Xs[:,i]
            end
            for i in (2, 1:150, 2:10, [2,5,7], [2,1])
                @test_throws BoundsError getobs(Xs, i, obsdim = 12)
                @test getobs(Xs,i) == Xs[:,i]
                @test getobs(ys,i) == ys[i]
                @test getobs(ys, i; obsdim = ndims(ys)) == ys[i]
                @test getobs(ys, i; obsdim = 1) == ys[i]
            end
        end
    
        @testset "Tuple" begin
            @test_throws DimensionMismatch getobs((X,yv), 1; obsdim=(2,))
            # bounds checking correctly
            @test_throws BoundsError getobs((X,y), 151)
            # special case empty tuple
            @test @inferred(getobs((), 10; obsdim = 1)) === ()
            @test @inferred(getobs((), 10)) === ()
            @test getobs((), 10; obsdim = 1) === ()
            tx, ty = getobs((Xv, yv), 10:50)
            @test typeof(tx) <: Array
            @test typeof(ty) <: Array
            for i in (1:150, 2:10, [2,5,7], [2,1])
                @test_throws DimensionMismatch getobs((X', y), i)
                @test_throws DimensionMismatch getobs((X, y),  i; obsdim=2)
                @test_throws DimensionMismatch getobs((X', y), i; obsdim=2)
                @test_throws DimensionMismatch getobs((X, y), i; obsdim=(1, 2))
                @test_throws DimensionMismatch getobs((X, y), i; obsdim=(2, 1, 1))
                @test_throws DimensionMismatch getobs((XX, X, y), i; obsdim=(2, 2, 1))
                @test_throws DimensionMismatch getobs((XX, X, y), i; obsdim=(3, 2))
                @test @inferred(getobs((X,y), i))  == (X[:,i], y[i])
                @test @inferred(getobs((X,yv), i)) == (X[:,i], y[i])
                @test @inferred(getobs((Xv,y), i)) == (X[:,i], y[i])
                @test @inferred(getobs((X,Y), i))  == (X[:,i], Y[:,i])
                @test @inferred(getobs((X,yt), i)) == (X[:,i], yt[:,i])
                @test @inferred(getobs((XX,X,y), i)) == (XX[:,:,i], X[:,i], y[i])
                @test getobs((XX,X,y), i, obsdim=(3,2,1)) == (XX[:,:,i], X[:,i], y[i])
                @test getobs((X, y), i, obsdim=(ndims(X), ndims(y)))  == (X[:,i], y[i])
                @test getobs((X',y), i, obsdim=1) == (X'[i,:], y[i])
                @test getobs((X,yv), i, obsdim=(ndims(X), ndims(yv)))  == (X[:,i], y[i])
                @test getobs((Xv,y), i, obsdim=(ndims(Xv), ndims(y)))  == (X[:,i], y[i])
                @test getobs((X, Y), i, obsdim=ndims(X))  == (X[:,i], Y[:,i])
                @test getobs((X',y), i, obsdim=1)  == (X'[i,:], y[i])
                @test getobs((X, y), i, obsdim=(2, 1))  == (X[:,i], y[i])
                @test getobs((X',y), i, obsdim=(1, 1))  == (X'[i,:], y[i])
                @test getobs((X',yt), i, obsdim=(1, 2))  == (X'[i,:], yt[:,i])
                @test getobs((X',yt), i, obsdim=(1, ndims(yt)))  == (X'[i,:], yt[:,i])
                # compare if obs match in tuple
                x1, y1 = getobs((X1,Y1), i)
                @test all(x1' .== y1)
                x1, y1, z1 = getobs((X1,Y1,sparse(X1)), i)
                @test all(x1' .== y1)
                @test all(x1 .== z1)
            end
            @test getobs((X,y), 2, obsdim=(ndims(X), ndims(y))) == (X[:,2], y[2])
            @test getobs((X,y), 2, obsdim=(2, 1)) == (X[:,2], y[2])
            @test getobs((Xv,y), 2) == (X[:,2], y[2])
            @test getobs((X,yv), 2) == (X[:,2], y[2])
            @test getobs((X,Y), 2) == (X[:,2], Y[:,2])
            @test getobs((XX,X,y), 2) == (XX[:,:,2], X[:,2], y[2])
        end
    end

    @testset "getobs!" begin
        @testset "Array and Subarray" begin
            Xbuf = similar(X)
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
            # string vector
            @test getobs!("setosa", y, 1) == "setosa"
            @test getobs!(nothing, y, 1) == "setosa"
        end

        @testset "SparseArray" begin
            ALLOWED_T = Union{Float64, SparseVector{Float64, Int64}}
            # Sparse Arrays opt-out of buffer usage
            @test @inferred(ALLOWED_T, getobs!(nothing, Xs, 1)) == getobs(Xs, 1)
            @test @inferred(ALLOWED_T, getobs!(nothing, Xs, 5:10)) == getobs(Xs, 5:10)
            @test @inferred(ALLOWED_T, getobs!(nothing, Xs, 2; obsdim=1)) == getobs(Xs, 2, obsdim=1)
            @test getobs!(nothing, Xs, 2, obsdim = 1) == getobs(Xs, 2, obsdim=1)
            @test @inferred(ALLOWED_T, getobs!(nothing, ys, 1)) === getobs(ys, 1)
            @test @inferred(ALLOWED_T, getobs!(nothing, ys, 5:10)) == getobs(ys, 5:10)
            @test @inferred(ALLOWED_T, getobs!(nothing, ys, 5:10; obsdim=1)) == getobs(ys, 5:10)
            @test getobs!(nothing, ys, 5:10, obsdim=1) == getobs(ys, 5:10)
        end
    
        @testset "Tuple" begin
            @test_throws MethodError getobs!((nothing,nothing), (X,y))
            @test getobs!((nothing,nothing), (X,y), 1:5) == getobs((X,y), 1:5)
            @test_throws MethodError getobs!((nothing,nothing,nothing), (X,y))
            xbuf = zeros(4,2)
            ybuf = ["foo", "bar"]
            @test_throws MethodError getobs!((xbuf,), (X,y))
            @test_throws MethodError getobs!((xbuf,ybuf,ybuf), (X,y))
            @test_throws DimensionMismatch getobs!((xbuf,), (X,y), 1:5)
            @test_throws DimensionMismatch getobs!((xbuf,ybuf,ybuf), (X,y), 1:5)
            @test @inferred(getobs!((xbuf,ybuf),(X,y), 2:3)) === (xbuf,ybuf)
            @test xbuf == getobs(X, 2:3)
            @test ybuf == getobs(y, 2:3)
            @test @inferred(getobs!((xbuf,ybuf),(X,y), [50,150])) === (xbuf,ybuf)
            @test xbuf == getobs(X, [50,150])
            @test ybuf == getobs(y, [50,150])
    
            xbuf2 = zeros(2,4)
            @test @inferred(getobs!((xbuf2,ybuf),(X',y), 4:5; obsdim = 1)) === (xbuf2,ybuf)
            @test xbuf2 == getobs(X', 4:5, obsdim=1)
            @test ybuf == getobs(y, 2:3)
    
            @test @inferred(getobs!((xbuf2,ybuf,xbuf),(X',y,X), 99:100, obsdim=(1, 1, 2))) === (xbuf2,ybuf,xbuf)
            getobs!(xbuf2, X', 99:100, obsdim=1)
            @test xbuf2 == getobs(X', 99:100, obsdim=1)
            @test ybuf  == getobs(y, 99:100)
            @test xbuf == getobs(X, 99:100)
    
            @test getobs!((xbuf2,ybuf,xbuf),(X',y,X), 9:10, obsdim=(1,1,2)) === (xbuf2,ybuf,xbuf)
            @test xbuf2 == getobs(X', 9:10, obsdim=1)
            @test ybuf  == getobs(y, 9:10)
            @test xbuf == getobs(X, 9:10)
    
            @test getobs!((nothing,xbuf),(Xs,X), 3:4) == (getobs(Xs,3:4),xbuf)
            @test xbuf == getobs(X,3:4)
        end
    
    end

    @testset "tuple" begin
        # A dataset with 3 observations, each with 2 input features
        X, Y = rand(2, 3), rand(3)
        dataset = (X, Y)
        ALLOWED_T = Tuple{Union{Float64, Vector{Float64}}, Float64}
        @test nobs(dataset) == 3
        if VERSION >= v"1.6"
            o = @inferred ALLOWED_T getobs(dataset, 2)
        else
            o = getobs(dataset, 2)
        end
        @test o[1] == X[:,2]
        @test o[2] == Y[2]

        if VERSION >= v"1.6"
            o = @inferred ALLOWED_T getobs(dataset, 1:2)
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
            ALLOWED_T = @NamedTuple{x::Union{Float64, Vector{Float64}}, y::Float64}
            o = @inferred ALLOWED_T getobs(dataset, 2)
        else
            o = getobs(dataset, 2)
        end
        @test o.x == X[:,2]
        @test o.y == Y[2]

        if VERSION >= v"1.6"
            ALLOWED_T = @NamedTuple{x::Union{Float64, Vector{Float64}}, y::Float64}
            o = @inferred ALLOWED_T getobs(dataset, 1:2)
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

@testset "AbstractDataContainer" begin
    struct TestContainer{T} <: LearnBase.AbstractDataContainer
        data::T
    end
    LearnBase.nobs(c::TestContainer; obsdim = LearnBase.default_obsdim(c)) = length(c.data)
    Base.length(c::TestContainer) = LearnBase.nobs(c)
    LearnBase.getobs(c::TestContainer, idx; obsdim = LearnBase.default_obsdim(c)) = c.data[idx]

    cont = TestContainer([2,4,6,8,10])
    @test collect(cont) == [2,4,6,8,10]
end
