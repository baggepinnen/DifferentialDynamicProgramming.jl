module DifferentialDynamicProgramming
using LinearAlgebra, Statistics, Printf
const DEBUG = false # Set this flag to true in order to print debug messages
# package code goes here

export QPTrace, boxQP, demoQP, Trace, iLQG, demo_linear

include("boxQP.jl")
include("iLQG.jl")
include("demo_linear.jl")
dir(paths...) = joinpath(@__DIR__, "..", paths...)
# @require ControlSystems include("system_pendcart.jl")

function debug(x)
    DEBUG && printstyled(string(x),"\n", color=:blue)
end

end # module
