module DifferentialDynamicProgramming
using LTVModelsBase, Requires, ValueHistories
const DEBUG = false # Set this flag to true in order to print debug messages
# package code goes here

export QPTrace, boxQP, demoQP, iLQG,iLQGkl, demo_linear, demo_linear_kl, GaussianPolicy


include("boxQP.jl")
include("iLQG.jl")
include("iLQGkl.jl")
include("forward_pass.jl")
include("backward_pass.jl")
include("demo_linear.jl")
@require ControlSystems include("system_pendcart.jl")

function debug(x)
    DEBUG && print_with_color(:blue, string(x),"\n")
end

end # module
