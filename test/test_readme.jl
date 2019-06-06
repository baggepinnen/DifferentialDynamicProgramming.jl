using Test, Statistics, LinearAlgebra, Random, DifferentialDynamicProgramming
# make stable linear dynamics
Random.seed!(0)
eye = DifferentialDynamicProgramming.eye
h    = .01  # time step
n    = 10   # state dimension
m    = 2    # control dimension
# control limits
lims = []# ones(m,1)*[-1 1]*.6

T    = 1000             # horizon
# quadratic costs
const Q    = SMatrix{n,n,Float64}(h*eye(n))
const R    = SMatrix{m,m,Float64}(.1*h*eye(m))

@time costs = map(1:10) do MCiteration
    A    = @SMatrix randn(n,n)
    A    = A-A' # skew-symmetric = pure imaginary eigenvalues
    A    = exp(h*A)        # discrete time
    B    = h* @SMatrix(randn(n,m))


    x0   = ones(n)        # initial state
    u0   = .1*randn(m,T) |> tosvec    # initial controls

    # optimization problem
    N    = T+1
    fx   = A
    fu   = B
    cxx  = Q
    cxu  = 0*B
    cuu  = R

    # Specify dynamics functions
    function lin_dyn_df(x,u,Q,R)
        cx  = Ref(Q) .* x
        cu  = Ref(R) .* u
        fxx=fxu=fuu = nothing
        return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
    end
    function lin_dyn_f(x,u,A,B)
        xnew = A*x + B*u
        return xnew
    end

    function lin_dyn_cost(x,u,Q)
        0.5*sum(x'Q*x + u'R*u for (x,u) in zip(x,u))
    end

    f(x,u,i)     = lin_dyn_f(x,u,A,B)
    costfun(x,u) = lin_dyn_cost(x,u,Q)
    df(x,u)      = lin_dyn_df(x,u,Q,R)
    # plotFn(x)  = plot(squeeze(x,2)')


    # run the optimization
    @time x, u, L, Vx, Vxx, cost, otrace = iLQG(f,costfun,df, x0, u0, lims=lims, verbosity=3);
    # using Plots
    # plot(x', title="States", subplot=1, layout=(3,1), show=true)
    # plot!(u', title="Control signals", subplot=2, show=true)
    # plot!(cost, title="Cost", subplot=3, show=true)

    sum(cost)
end

@test maximum(costs) < 25 # This should be the case most of the times
@test mean(costs) < 10 # This should be the case most of the times
@test minimum(costs) < 5 # This should be the case most of the times



# 8.136281 seconds (13.75 M allocations: 2.661 GiB, 13.57% gc time)
# 2.174562 seconds (2.90 M allocations: 157.990 MiB, 5.06% gc time)
