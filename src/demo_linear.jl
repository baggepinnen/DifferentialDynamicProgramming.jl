
function demo_linear()
    println("Running linear demo function for DifferentialDynamicProgramming.jl")
    function lin_dyn_f(x,u,A,B,Q,R)
        u[isnan(u)] = 0
        f = A*x + B*u
        c = 0.5*sum(x.*(Q*x),1) + 0.5*sum(u.*(R*u),1)
        return f,c
    end

    function lin_dyn_fT(x,Q)
        c = 0.5*sum(x.*(Q*x),1)
        return c
    end
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
        u[isnan(u)] = 0
        cx  = Q*x
        cu  = R*u
        fxx=fxu=fuu = []
        return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
    end


    f(x,u,i)   = lin_dyn_f(x,u,A,B,Q,R)
    fT(x)      = lin_dyn_fT(x,Q)
    df(x,u,i)  = lin_dyn_df(x,u,Q,R)
    # plotFn(x)  = plot(squeeze(x,2)')

    # run the optimization
    @time x, u, L, Vx, Vxx, cost, otrace = iLQG(f,fT,df, x0, u0, lims=lims, plotFn= x -> 0 );

    totalcost = [ otrace[i].cost for i=1:length(otrace)]
    iters = sum(totalcost .> 0)
    totalcost = [ otrace[i].cost for i=1:iters]

    @require Plots begin
    Plots.plot(x')
    Plots.plot(cost',"k",linewidth=3, title="State Trajectories and cost", xlabel="Time step")
    Plots.plot(u',title="Control signals")
    Plots.plot(totalcost,title="Total cost")

end

end
