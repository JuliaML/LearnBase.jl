@testset "Other" begin
    @test LearnBase.Minimizable <: Any
    @test LearnBase.Transformation <: Any
    @test LearnBase.StochasticTransformation <: LearnBase.Transformation

    @test typeof(LearnBase.transform) <: Function
    @test typeof(LearnBase.transform!) <: Function
    @test typeof(LearnBase.learn) <: Function
    @test typeof(LearnBase.learn!) <: Function
    @test typeof(LearnBase.update) <: Function
    @test typeof(LearnBase.update!) <: Function

    @test typeof(LearnBase.grad) <: Function
    @test typeof(LearnBase.grad!) <: Function
    @test typeof(LearnBase.prox) <: Function
    @test typeof(LearnBase.prox!) <: Function

    # IntervalSet
    let s = LearnBase.IntervalSet(-1,1)
        @test typeof(s) == LearnBase.IntervalSet{Int}
        @test typeof(s) <: AbstractSet
        for x in (-1,0,0.5,1,1.0)
            @test x in s
        end
        for x in (-1-1e-10, 1+1e-10, -Inf, Inf, 2, NaN)
            @test !(x in s)
        end
        for i=1:10
            x = rand(s)
            @test typeof(x) == Float64
            @test x in s
        end
        xs = rand(s, 10)
        @test typeof(xs) == Vector{Float64}
        for x in xs
            @test typeof(x) == Float64
            @test x in s
        end
        @test LearnBase.randtype(s) == Float64
        # @show s LearnBase.randtype(s)
    end
    let s = LearnBase.IntervalSet(-1,1.0)
        @test typeof(s) == LearnBase.IntervalSet{Float64}
        @test typeof(s) <: AbstractSet
        @test 1 in s
        # @show s LearnBase.randtype(s)
        @test length(s) == 1
    end

    # IntervalSet{Vector}
    let s = LearnBase.IntervalSet([-1.,0.], [1.,1.])
        @test typeof(s) == LearnBase.IntervalSet{Vector{Float64}}
        @test typeof(s) <: AbstractSet
        @test LearnBase.randtype(s) == Vector{Float64}
        @test typeof(rand(s)) == Vector{Float64}
        @test rand(s) in s
        @test [-1, 0] in s
        @test !([-1.5,0] in s)
        @test !([0,2] in s)
        @test length(s) == 2
    end

    # DiscreteSet
    let s = LearnBase.DiscreteSet([-1,1])
        @test typeof(s) == LearnBase.DiscreteSet{Vector{Int}}
        @test typeof(s) <: AbstractSet
        for x in (-1, 1, -1.0, 1.0)
            @test x in s
        end
        for x in (0, Inf, -Inf, NaN)
            @test !(x in s)
        end
        for i=1:10
            x = rand(s)
            @test typeof(x) == Int
            @test x in s
        end
        xs = rand(s, 10)
        @test typeof(xs) == Vector{Int}
        for x in xs
            @test typeof(x) == Int
            @test x in s
        end
        @test LearnBase.randtype(s) == Int
        @test length(s) == 2
        @test s[1] == -1
    end
    let s = LearnBase.DiscreteSet([-1,1.0])
        @test typeof(s) == LearnBase.DiscreteSet{Vector{Float64}}
        @test typeof(s) <: AbstractSet
        @test typeof(rand(s)) == Float64
        @test typeof(rand(s, 2)) == Vector{Float64}
    end

    # TupleSet
    let s = LearnBase.TupleSet(LearnBase.IntervalSet(0,1), LearnBase.DiscreteSet([0,1]))
        @test typeof(s) == LearnBase.TupleSet{Tuple{LearnBase.IntervalSet{Int}, LearnBase.DiscreteSet{Vector{Int}}}}
        @test typeof(s) <: AbstractSet
        for x in ([0,0], [0.0,0.0], [0.5,1.0])
            @test x in s
        end
        for x in ([0,0.5], [-1,0])
            @test !(x in s)
        end
        @test typeof(rand(s)) == Vector{Float64}
        @test typeof(rand(s, 2)) == Vector{Vector{Float64}}
        @test typeof(rand(s, Tuple)) == Tuple{Float64,Int}
        @test typeof(rand(s, Tuple, 2)) == Vector{Tuple{Float64,Int}}
        @test LearnBase.randtype(s) == Vector{Float64}

        tot = 0
        for (i,x) in enumerate(s)
            @test x == s.sets[i]
            tot += length(x)
        end
        @test length(s) == tot
    end

    # arrays of sets
    let s = [LearnBase.IntervalSet(0,1), LearnBase.DiscreteSet([0,1])]
        @test typeof(s) == Vector{AbstractSet}
        for x in ([0,0], [0.0,0.0], [0.5,1.0])
            @test x in s
        end
        for x in ([0,0.5], [-1,0])
            @test !(x in s)
        end
        @test typeof(rand(s)) == Vector{Float64}
        @test typeof(rand(s, 2)) == Vector{Vector{Float64}}
    end
end
