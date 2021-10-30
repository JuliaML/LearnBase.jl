module LearnBase

import StatsBase: nobs

# AGGREGATION MODES
include("aggmode.jl")

# OBSERVATION DIMENSIONS
include("observation.jl")

# LABEL ENCONDINGS
include("labels.jl")

# LEARNING COSTS (e.g. loss & penalty)
include("costs.jl")

# OTHER CONCEPTS
include("other.jl")

end # module
