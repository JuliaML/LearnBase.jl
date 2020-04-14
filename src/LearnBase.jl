module LearnBase

using Markdown # for docstrings

# AGGREGATION MODES
include("aggmode.jl")

# OBSERVATION DIMENSIONS
include("obsdim.jl")

# LEARNING COSTS (e.g. loss & penalty)
include("costs.jl")

# OTHER CONCEPTS
include("other.jl")

end # module
