plotstuff_linear(args...) = println("Install package Plots.jl (and call using Plots) to plot results in the end of demo_linear")



function demo_linear(;kwargs...)
    println("Running linear demo function for DifferentialDynamicProgramming.jl")

    # make stable linear dynamics
    h = .01         # time step
    n = 10          # state dimension
    m = 2           # control dimension
    A = randn(n,n)
    A = A-A'        # skew-symmetric = pure imaginary eigenvalues
    A = exp(h*A)   # discrete time
    B = h*randn(n,m)

    # quadratic costs
    Q    = h*eye(n)
    R    = .1*h*eye(m)

    # control limits
    lims = []            #ones(m,1)*[-1 1]*.6

    T    = 1000          # horizon
    x0   = ones(n)     # initial state
    u0   = .1*randn(m,T) |> tosvec # initial controls

    # optimization problem
    N    = T+1
    fx   = A
    fu   = B
    cxx  = Q
    cxu  = zeros(size(B))
    cuu  = R
    function lin_dyn_df(x,u,Q,R)
        cx  = Ref(Q) .* x
        cu  = Ref(R) .* u
        fxx=fxu=fuu = nothing
        return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
    end
    function lin_dyn_f(x,u,A,B,Q,R)
        A*x + B*u
    end
    quadform(x::AbstractVector{<:AbstractVector},Q) = sum(x'Q*x for x in x)
    quadform(x,Q) = x'Q*x
    lin_dyn_fT(x,Q) = 0.5*quadform(x,Q)
    f(x,u,i)        = lin_dyn_f(x,u,A,B,Q,R)
    costfun(x,u)    = 0.5*quadform(x,Q) + 0.5*quadform(u,R)
    df(x,u)         = lin_dyn_df(x,u,Q,R)
    # plotFn(x)  = plot(squeeze(x,2)')

    # run the optimization
    @time x, u, traj_new, Vx, Vxx, cost, otrace = iLQG(f,costfun,df, x0, u0; lims=lims,kwargs...);

    totalcost = get(otrace, :cost)[2]
    plotstuff_linear(x,u,[cost],totalcost)
    x, u, traj_new, Vx, Vxx, cost, otrace
end


function demo_linear_kl(;kwargs...)
    println("Running linear demo function with KL-divergence constraint for DifferentialDynamicProgramming.jl")

    # make stable linear dynamics
    h    = .01           # time step
    n    = 10            # state dimension
    m    = 2             # control dimension
    A    = randn(n,n)
    A    = A-A'          # skew-symmetric = pure imaginary eigenvalues
    A    = exp(h*A)     # discrete time
    B    = h*randn(n,m)

    # quadratic costs
    Q    = h*eye(n)
    R    = .1*h*eye(m)

    # control limits
    lims = []            #ones(m,1)*[-1 1]*.6

    T    = 1000          # horizon
    x0   = ones(n)       # initial state
    u    = .1*randn(m,T) |> tosvec # initial controls

    # optimization problem
    N    = T+1
    fx   = [tosmat(A) for _ in 1:T]
    fu   = [tosmat(B) for _ in 1:T]
    cxx  = [tosmat(Q) for _ in 1:T]
    cxu  = [tosmat(zeros(size(B))) for _ in 1:T]
    cuu  = [tosmat(R) for _ in 1:T]
    function lin_dyn_df(x,u,Q,R)
        cx  = Ref(Q) .* x
        cu  = Ref(R) .* u
        fxx=fxu=fuu = nothing
        return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
    end
    function lin_dyn_f(x,u,A,B,Q,R)
        A*x + B*u
    end
    quadform(x::AbstractVector{<:AbstractVector},Q) = sum(x'Q*x for x in x)
    quadform(x,Q) = x'Q*x

    dyn = (x,u,i)   -> lin_dyn_f(x,u,A,B,Q,R)
    costf = (x,u) -> 0.5*quadform(x,Q) + 0.5*quadform(u,R)
    diffdyn = (x,u)  -> lin_dyn_df(x,u,Q,R)

    function rollout(u)
        x = zeros(n,T) |> tosvec
        @show typeof(x)
        x[1] = x0
        for t = 1:T-1
            x[t+1] = dyn(x[t],u[t],t)
        end
        x
    end
    x = rollout(u)
    model = LinearTimeVaryingModelsBase.SimpleLTVModel(repeat(A,1,1,N),repeat(B,1,1,N),false)
    # plotFn(x)  = plot(squeeze(x,2)')
    traj = GaussianPolicy(Float64,T,n,m)
    # run the optimization
    local Vx, Vxx, cost, otrace, totalcost
    outercosts = zeros(5)
    @time for iter = 1:5
        cost0 = 0.5*(quadform(x,Q) +quadform(u,R))
        x, u, traj, Vx, Vxx, cost, otrace = iLQGkl(dyn,costf,diffdyn, x, traj, model; cost=cost0, lims=lims,kwargs...);
        totalcost = get(otrace, :cost)[2]
        outercosts[iter] = sum(totalcost)
        println("Outer loop: Cost = ", sum(cost))
    end

    totalcost = get(otrace, :cost)[2]
    plotstuff_linear(x,u,[cost],min.(totalcost,400))
    # plotstuff_linear(x,u,totalcost,outercosts)
    x, u, traj, Vx, Vxx, cost, otrace
end
