module DifferentialDynamicProgramming
using Requires
const DEBUG = false
# package code goes here

export QPTrace, boxQP, demoQP, Trace, iLQG, demo_linear

include("boxQP.jl")
include("iLQG.jl")
include("demo_linear.jl")
# include("system_pendcart.jl")

function debug(x)
    DEBUG && print_with_color(:blue, string(x),"\n")
end

end # module
