plotstuff_pendcart(args...) = println("Install package Plots.jl (and call using Plots) to plot results in the end of demo_pendcart")

function care(A, B, Q, R)
    G = try
        B*inv(R)*B'
    catch
        error("R must be non-singular.")
    end
    Z = [A  -G;
    -Q  -A']

    S = schur(Z)
    S = ordschur(S, real(S.values).<0)
    U = S.Z

    (m, n) = size(U)
    U11 = U[1:div(m, 2), 1:div(n,2)]
    U21 = U[div(m,2)+1:m, 1:div(n,2)]
    return U21/U11
end
function lqr(A, B, Q, R)
    S = care(A, B, Q, R)
    K = R\B'*S
    return K
end


"""
    demo_pendcart(;kwargs...)

Run the iLQG Function to find an optimal trajectory For the "pendulum on a cart system". Requires package ControlSystems.jl

# Arguments
`x0     = [π-0.6,0,0,0]`
`goal   = [π,0,0,0]`
`Q      = Diagonal([10,1,2,1])` : State weight matrix
`R      = 1`                    : Control weight matrix
`lims   = 5.0*[-1 1]`           : control limits,
`T      = 600`                  : Number of time steps
`doplot = true`                 : Plot results
"""
function demo_pendcart(;x0 = [π-0.6,0,0,0], goal = [π,0,0,0],
    Q      = Diagonal([10.,1,2,1]), # State weight matrix
    R      = 1.,                    # Control weight matrix
    lims   = 5.0*[-1 1],            # control limits,
    T      = 600,                   # Number of time steps
    doplot = true                   # Plot results
    )

    N = T+1
    g = 9.82
    l = 0.35 # Length of pendulum
    h = 0.01 # Sample time
    d = 0.99
    A = [0 1 0 0; # Linearlized system dynamics matrix, continuous time
    g/l -d 0 0;
    0 0 0 1;
    0 0 0 0]
    B = [0, -1/l, 0, 1]
    C = eye(4) # Assume all states are measurable
    D = 4
    L = lqr(A,B,Q,R) # Calculate the optimal state feedback
    I = T

    function fsys_closedloop(t,x,L,xd)
        dx = copy(x)
        dx[1] -= pi
        u = -(L*dx)[1]
        xd[1] = x[2]
        xd[2] = -g/l * sin(x[1]) + u/l * cos(x[1]) - d*x[2]
        xd[3] = x[4]
        xd[4] = u
    end

    function fsys(t,x,u,xd)
        xd[1] = x[2]
        xd[2] = -g/l * sin(x[1]) + u/l * cos(x[1]) - d*x[2]
        xd[3] = x[4]
        xd[4] = u
    end

    dfvec = zeros(4)
    function dfsys(x,u)
        dfvec[1] = x[1]+h*x[2]
        dfvec[2] = x[2]+h*(-g/l*sin(x[1])+u[1]/l*cos(x[1])- d*x[2])
        dfvec[3] = x[3]+h*x[4]
        dfvec[4] = x[4]+h*u[1]
        dfvec
    end


    function cost_quadratic(x,u)
        local d = (x.-goal)
        0.5(d'*Q*d + u'R*u)[1]
    end

    function cost_quadratic(x::Matrix,u)
        local d = (x.-goal)
        T = size(u,2)
        c = Vector{Float64}(undef,T+1)
        for t = 1:T
            c[t] = 0.5(d[:,t]'*Q*d[:,t] + u[:,t]'R*u[:,t])[1]
        end
        c[end] = cost_quadratic(x[:,end][:],[0.0])
        return c
    end

    cx = zeros(4,T)
    cu = zeros(1,T)
    cxu = zeros(D,1)

    function dcost_quadratic(x,u)
        cx  .= Q*(x.-goal)
        cu  .= R.*u
        return cx,cu,cxu
    end


    function lin_dyn_f(x,u,i)
        u[isnan.(u)] .= 0
        f = dfsys(x,u)
    end


    fxc           = Array{Float64}(undef,D,D,I)
    fuc           = Array{Float64}(undef,D,1,I)
    fxd           = Array{Float64}(undef,D,D,I)
    fud           = Array{Float64}(undef,D,1,I)
    for ii = 1:I
        fxc[:,:,ii] = [0 1 0 0;
        0 0 0 0;
        0 0 0 1;
        0 0 0 0]
        fuc[:,:,ii] = [0, 0, 0, 1]
    end

    function lin_dyn_df(x,u)
        u[isnan.(u)] .= 0
        D            = size(x,1)
        nu,I         = size(u)
        cx,cu,cxu    = dcost_quadratic(x,u)
        cxx          = Q
        cuu          = [R]
        for ii = 1:I
            fxc[2,1,ii] = -g/l*cos(x[1,ii])-u[ii]/l*sin(x[1,ii])
            fxc[2,2,ii] = -d
            fuc[2,1,ii] = cos(x[1,ii])/l
            ABd = exp([fxc[:,:,ii]*h  fuc[:,:,ii]*h; zeros(nu, D + nu)])# ZoH sampling
            fxd[:,:,ii] = ABd[1:D,1:D]
            fud[:,:,ii] = ABd[1:D,D+1:D+nu]
        end
        fxx=fxu=fuu = []
        return fxd,fud,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
    end

    x = zeros(4,N)
    u = zeros(1,T)

    """
    Simulate a pendulum on a cart using the non-linear equations
    """
    function simulate_pendcart(x0,L, dfsys, cost)
        x[:,1] = x0
        u[1] = 0
        for t = 2:T
            dx     = copy(x[:,t-1])
            dx[1] -= pi
            u[t]   = -(L*dx)[1]
            if !isempty(lims)
                u[t]   = clamp(u[t],lims[1],lims[2])
            end
            x[:,t] = dfsys(x[:,t-1],u[t])
        end
        dx      = copy(x[:,T])
        dx[1]  -= pi
        uT      = -(L*dx)[1]
        if !isempty(lims)
            uT   = clamp(uT,lims[1],lims[2])
        end
        x[:,T+1] = dfsys(x[:,T],uT)
        c = cost(x,u)

        return x, u, c
    end


    # Simulate the closed loop system with regular LQG control and watch it fail due to control limits
    x00, u00, cost00 = simulate_pendcart(x0, L, dfsys, cost_quadratic)


    f(x,u,i) = lin_dyn_f(x,u,i)
    df(x,u)  = lin_dyn_df(x,u)
    # plotFn(x)  = plot(squeeze(x,2)')

    println("Entering iLQG function")
    # subplot(n=4,nc=2)
    x, u, L, Vx, Vxx, cost, trace = iLQG(f,cost_quadratic, df, x0, 0*u00,
    lims      = lims,
    # plotFn  = x -> Plots.subplot!(x'),
    regType   = 2,
    α         = exp10.(range(0.2, stop=-3, length=6)),
    λmax      = 1e15,
    verbosity = 3,
    tol_fun   = 1e-8,
    tol_grad   = 1e-8,
    max_iter  = 1000);

    doplot && plotstuff_pendcart(x00, u00, x,u,cost00,cost,trace)
    println("Done")

    return x, u, L, Vx, Vxx, cost, trace
end
