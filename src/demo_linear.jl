
import Plots
function plotstuff_linear(x,u,cost,totalcost)
    p = Plots.plot(layout=(2,2))
    Plots.plot!(p,x', title="State Trajectories", xlabel="Time step",legend=false, subplot=1, show=false)
    Plots.plot!(p,cost,c=:black,linewidth=3, title="Cost", xlabel="Time step", subplot=2, show=false)
    Plots.plot!(p,u',title="Control signals", xlabel="Time step", subplot=3, show=false)
    Plots.plot!(p,totalcost,title="Total cost", xlabel="Iteration", subplot=4, show=false)
    Plots.gui()
end


function demo_linear(;kwargs...)
    println("Running linear demo function for DifferentialDynamicProgramming.jl")

    # make stable linear dynamics
    h = .01         # time step
    n = 10          # state dimension
    m = 2           # control dimension
    A = randn(n,n)
    A = A-A'        # skew-symmetric = pure imaginary eigenvalues
    A = expm(h*A)   # discrete time
    B = h*randn(n,m)

    # quadratic costs
    Q = h*eye(n)
    R = .1*h*eye(m)

    # control limits
    lims = [] #ones(m,1)*[-1 1]*.6

    T        = 1000              # horizon
    x0       = ones(n,1)        # initial state
    u0       = .1*randn(m,T)     # initial controls

    # optimization problem
    N   = T+1
    fx  = A
    fu  = B
    cxx = Q
    cxu = zeros(size(B))
    cuu = R

    function lin_dyn_df(x,u,Q,R)
        u[isnan.(u)] = 0
        cx  = Q*x
        cu  = R*u
        fxx=fxu=fuu = []
        return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
    end
    function lin_dyn_f(x,u,A,B,Q,R)
        u[isnan.(u)] = 0
        f = A*x + B*u
        c = 0.5*sum(x.*(Q*x)) + 0.5*sum(u.*(R*u))
        return f,c
    end

    function lin_dyn_fT(x,Q)
        c = 0.5*sum(x.*(Q*x))
        return c
    end

    f(x,u,i)   = lin_dyn_f(x,u,A,B,Q,R)
    fT(x)      = lin_dyn_fT(x,Q)
    df(x,u,i)  = lin_dyn_df(x,u,Q,R)
    # plotFn(x)  = plot(squeeze(x,2)')

    # run the optimization
    @time x, u, L, Vx, Vxx, cost, otrace = iLQG(f,fT,df, x0, u0; lims=lims, plotFn= x -> 0 ,kwargs...);

    totalcost = [ t.cost for t in otrace]
    iters = sum(totalcost .> 0)
    totalcost = [ otrace[i].cost for i=1:iters]


    plotstuff_linear(x,u,cost,totalcost)

end
