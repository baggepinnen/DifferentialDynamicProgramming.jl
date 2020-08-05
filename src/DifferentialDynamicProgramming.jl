module DifferentialDynamicProgramming
using LinearTimeVaryingModelsBase, Requires, ValueHistories, LinearAlgebra, Statistics, Printf, StaticArrays
using MacroTools: @capture, postwalk, prewalk
const DEBUG = false # Set this flag to true in order to print debug messages


export QPTrace, boxQP, demoQP, iLQG,iLQGkl, demo_linear, demo_linear_kl, demo_pendcart, GaussianPolicy, tosvec

tosvec(x::AbstractMatrix{<:Number}) = reinterpret(SVector{size(x,1),eltype(x)}, x)[:]
tosvec(x::AbstractVector{<:Number}) = SVector{length(x),eltype(x)}(x)
tosmat(x::AbstractMatrix{<:Number}) = SMatrix{size(x,1),size(x,2),eltype(x)}(x)
eye(n) = Matrix{Float64}(I,n,n)
Base.strides(::SArray{Tuple{i,j}}) where {i,j} = (1,i)

function __init__()
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
    @eval LinearAlgebra.adjoint(x::String) = x
    @eval function plotstuff_linear(x,u,cost,totalcost)
        p = Plots.plot(layout=(2,2))
        Plots.plot!(p,reduce(hcat,x)', title="State Trajectories", xlabel="Time step",legend=false, subplot=1, show=false)
        Plots.hline!(p,cost,c=:black,linewidth=3, title="Cost", xlabel="Time step", subplot=2, show=false)
        Plots.plot!(p,reduce(hcat,u)',title="Control signals", xlabel="Time step", subplot=3, show=false)
        Plots.plot!(p,totalcost,title="Total cost", xlabel="Iteration", subplot=4, show=false)
        Plots.gui()
    end
    @eval function plotstuff_pendcart(x00, u00, x,u,cost00,cost,otrace)
        cp = Plots.plot(layout=(1,3))
        sp = Plots.plot(reduce(hcat,x00)',title=["\$x_$(i)\$" for i=1:length(x00[1])]', lab="Simulation", layout=(2,2))
        Plots.plot!(sp,reduce(hcat,x)', title=["\$x_$(i)\$" for i=1:length(x00[1])]', lab="Optimized", xlabel="Time step", legend=true)

        Plots.plot!(cp,cost00, title="Cost", lab="Simulation", subplot=2)
        Plots.plot!(cp,reduce(hcat,u)', legend=true, title="Control signal",lab="Optimized", subplot=1)
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
