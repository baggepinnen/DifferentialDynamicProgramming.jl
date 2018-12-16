module DifferentialDynamicProgramming
using LinearTimeVaryingModelsBase, Requires, ValueHistories, LinearAlgebra, Statistics, Printf
const DEBUG = false # Set this flag to true in order to print debug messages
# package code goes here

export QPTrace, boxQP, demoQP, iLQG,iLQGkl, demo_linear, demo_linear_kl, demo_pendcart, GaussianPolicy


eye(n) = Matrix{Float64}(I,n,n)

function __init__()
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
    @eval LinearAlgebra.adjoint(x::String) = x
    @eval function plotstuff_linear(x,u,cost,totalcost)
        p = Plots.plot(layout=(2,2))
        Plots.plot!(p,x', title="State Trajectories", xlabel="Time step",legend=false, subplot=1, show=false)
        Plots.plot!(p,cost,c=:black,linewidth=3, title="Cost", xlabel="Time step", subplot=2, show=false)
        Plots.plot!(p,u',title="Control signals", xlabel="Time step", subplot=3, show=false)
        Plots.plot!(p,totalcost,title="Total cost", xlabel="Iteration", subplot=4, show=false)
        Plots.gui()
    end
    @eval function plotstuff_pendcart(x00, u00, x,u,cost00,cost,otrace)
        cp = Plots.plot(layout=(1,3))
        sp = Plots.plot(x00',title=["\$x_$(i)\$" for i=1:size(x00,1)]', lab="Simulation", layout=(2,2))
        Plots.plot!(sp,x', title=["\$x_$(i)\$" for i=1:size(x00,1)]', lab="Optimized", xlabel="Time step", legend=true)

        Plots.plot!(cp,cost00, title="Cost", lab="Simulation", subplot=2)
        Plots.plot!(cp,u', legend=true, title="Control signal",lab="Optimized", subplot=1)
        Plots.plot!(cp,cost[2:end], legend=true, title="Cost",lab="Optimized", xlabel="Time step", subplot=2, yscale=:log10)
        iters = sum(cost .> 0)
        filter!(x->x>0,cost)
        Plots.plot!(cp, get(otrace, :cost)[2], yscale=:log10,xscale=:log10, title="Total cost", xlabel="Iteration", legend=false, subplot=3)
        Plots.plot(sp,cp)
        Plots.gui()
    end
end
end

dir(paths...) = joinpath(@__DIR__, "..", paths...)
include("boxQP.jl")
include("iLQG.jl")
include("iLQGkl.jl")
include("forward_pass.jl")
include("backward_pass.jl")
include("demo_linear.jl")
include("system_pendcart.jl")

function debug(x)
    DEBUG && printstyled(string(x),"\n", color=:blue)
end

end # module
