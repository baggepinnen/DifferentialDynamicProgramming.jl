using Base.Test
# make stable linear dynamics
costs = map(1:10) do MCiteration
    h    = .01  # time step
    n    = 10   # state dimension
    m    = 2    # control dimension
    A    = randn(n,n)
    A    = A-A' # skew-symmetric = pure imaginary eigenvalues
    A    = expm(h*A)        # discrete time
    B    = h*randn(n,m)

    # quadratic costs
    Q    = h*eye(n)
    R    = .1*h*eye(m)

    # control limits
    lims = []# ones(m,1)*[-1 1]*.6

    T    = 1000             # horizon
    x0   = ones(n,1)        # initial state
    u0   = .1*randn(m,T)    # initial controls

    # optimization problem
    N    = T+1
    fx   = A
    fu   = B
    cxx  = Q
    cxu  = zeros(size(B))
    cuu  = R

    # Specify dynamics functions
    function lin_dyn_df(x,u,Q,R)
        u[isnan.(u)] = 0
        cx  = Q*x
        cu  = R*u
        fxx=fxu=fuu = []
        return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
    end
    function lin_dyn_f(x,u,A,B)
        u[isnan.(u)] = 0
        xnew = A*x + B*u
        return xnew
    end

    function lin_dyn_cost(x,u,Q)
        c = 0.5*sum(x.*(Q*x)) + 0.5*sum(u.*(R*u))
        return c
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
