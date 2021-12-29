module LearnBase

# AGGREGATION MODES
include("aggmode.jl")

# OBSERVATION DIMENSIONS
export default_obsdim, getobs, getobs!
include("observation.jl")

# LEARNING COSTS (e.g. loss & penalty)
include("costs.jl")

# OTHER CONCEPTS
include("other.jl")

end # module
