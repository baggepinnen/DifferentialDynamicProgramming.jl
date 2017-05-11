module DifferentialDynamicProgramming
using Requires
const DEBUG = false # Set this flag to true in order to print debug messages
# package code goes here

export QPTrace, boxQP, demoQP, Trace, iLQG,iLQGkl, demo_linear, demo_linear_kl
export AbstractModel, AbstractCost, ModelAndCost, define_modelcost_functions, cost, cost_final, dc, fit_model, predict, df

include("interfaces.jl")
include("boxQP.jl")
include("iLQG.jl")
include("iLQGkl.jl")
include("backward_pass.jl")
include("demo_linear.jl")
@require ControlSystems include("system_pendcart.jl")

function debug(x)
    DEBUG && print_with_color(:blue, string(x),"\n")
end

end # module
