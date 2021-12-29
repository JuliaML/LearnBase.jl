module LearnBase

using StatsBase: nobs

# AGGREGATION MODES
include("aggmode.jl")

# VIEW AND ITERATORS
include("iteration.jl")

# OBSERVATION DIMENSIONS
include("observation.jl")

# LEARNING COSTS (e.g. loss & penalty)
include("costs.jl")

# OTHER CONCEPTS
include("other.jl")

end # module
