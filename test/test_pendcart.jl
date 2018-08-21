using ControlSystems
function fsys_closedloop(t,x,L,xd)
    dx = copy(x)
    dx[1] -= pi
    u = -(L*dx)[1]
    xd[1] = x[2]
    xd[2] = -g/l * sin(x[1]) + u/l * cos(x[1])
    xd[3] = x[4]
    xd[4] = u
end

function fsys(t,x,u,xd)
    xd[1] = x[2]
    xd[2] = -g/l * sin(x[1]) + u/l * cos(x[1])
    xd[3] = x[4]
    xd[4] = u
end

function dfsys(x,u)
    [x[1]+h*x[2]; x[2]+h*(-g/0.35*sin(x[1])+u/0.35*cos(x[1])); x[3]+h*x[4]; x[4]+h*u]
end


function cost_quadratic(x,u)
    d = (x.-goal)
    0.5(d'*Q*d + u'R*u)[1]
end

function cost_quadratic(x::Matrix,u)
    d = (x.-goal)
    T = size(u,2)
    c = Vector{Float64}(T+1)
    for t = 1:T
        c[t] = 0.5(d[:,t]'*Q*d[:,t] + u[:,t]'R*u[:,t])[1]
    end
    c[end] = cost_quadratic(x[:,end][:],[0.0])
    return c
end

function dcost_quadratic(x,u)
    cx  = Q*(x.-goal)
    cu  = R.*u
    cxu = zeros(D,1)
    return cx,cu,cxu
end


function lin_dyn_f(x,u,i)
    u[isnan(u)] = 0
    f = dfsys(x,u)
    c = cost_quadratic(x,u)
    return f,c
end

function lin_dyn_fT(x)
    cost_quadratic(x,0.0)
end


function lin_dyn_df(x,u)
    u[isnan(u)] = 0
    D = size(x,1)
    nu,I = size(u)
    fx = Array{Float64}(D,D,I)
    fu = Array{Float64}(D,1,I)
    cx,cu,cxu = dcost_quadratic(x,u)
    cxx = Q
    cuu = [R]
    for ii = 1:I
        fx[:,:,ii] = [0 1 0 0;
        -g/0.35*cos(x[1,ii])-u[ii]/0.35*sin(x[1,ii]) 0 0 0;
        0 0 0 1;
        0 0 0 0]
        fu[:,:,ii] = [0, cos(x[1,ii])/0.35, 0, 1]
        ABd = exp([fx[:,:,ii]*h  fu[:,:,ii]*h; zeros(nu, D + nu)])# ZoH sampling
        fx[:,:,ii] = ABd[1:D,1:D]
        fu[:,:,ii] = ABd[1:D,D+1:D+nu]
    end
    fxx=fxu=fuu = []
    return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
end


"""
Simulate a pendulum on a cart using the non-linear equations
"""
function simulate_pendcart(x0,L, dfsys, cost)
    x = zeros(4,N)
    u = zeros(1,T)
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


T    = 600 # Number of time steps
N    = T+1
g    = 9.82
l    = 0.35 # Length of pendulum
h    = 0.01 # Sample time
lims = []#5.0*[-1 1] # control limits, e.g. ones(m,1)*[-1 1]*.6
goal = [π,0,0,0] # Reference point
A    = [0 1 0 0; # Linearlized system dynamics matrix, continuous time
g/l 0 0 0;
0 0 0 1;
0 0 0 0]
B   = [0, -1/l, 0, 1]
C   = eye(4) # Assume all states are measurable
D   = 4

sys = ss(A,B,C,zeros(4))
Q   = h*diagm([10,1,2,1]) # State weight matrix
R   = h*1 # Control weight matrix
L   = lqr(sys,Q,R) # Calculate the optimal state feedback

x0 = [π-0.6,0,0,0]


# Simulate the closed loop system with regular LQG control and watch it fail due to control limits
x00, u00, cost00 = simulate_pendcart(x0, L, dfsys, cost_quadratic)
u0 = 0u00

fx  = A
fu  = B
cxx = Q
cxu = zeros(size(B))
cuu = R

f(x,u,i) = lin_dyn_f(x,u,i)
fT(x)    = lin_dyn_fT(x)
df(x,u)  = lin_dyn_df(x,u)
# plotFn(x)  = plot(squeeze(x,2)')

# run the optimization
println("Entering iLQG function")
# subplot(n=4,nc=2)


alpha        = exp10.(range(0, stop=-3, length=11))
tol_fun      = 1e-7
tol_grad     = 1e-4
max_iter     = 500
λ            = 1
dλ           = 1
λfactor      = 1.6
λmax         = 1e10
λmin         = 1e-6
regType     = 1
reduce_ratio_min = 0
diff_fun     = -
plot         = 1
verbosity    = 2
plot_fun     = x->0
cost         = []
kl_step      = 0
traj_prev    = 0


regType=2
alpha= exp10.(range(0.2, stop=-3, length=6))
verbosity=3
tol_fun = 1e-7
max_iter=1000


import Base: length
EmptyMat3 = Array(Float64,0,0,0)
EmptyMat2 = Array(Float64,0,0)
emptyMat3(P) = Array(P,0,0,0)
emptyMat2(P) = Array(P,0,0)



mutable struct GaussianDist{P}
    T::Int
    n::Int
    m::Int
    fx::Array{P,3}
    fu::Array{P,3}
    Σ::Array{P,3}
    μx::Array{P,2}
    μu::Array{P,2}
end

GaussianDist(P) = GaussianDist(0,0,0,emptyMat3(P),emptyMat3(P),emptyMat3(P),emptyMat2(P),emptyMat2(P))
Base.isempty(gd::GaussianDist) = gd.T == gd.n == gd.m == 0
mutable struct GaussianTrajDist{P}
    policy::GaussianDist{P}
    dynamics::GaussianDist{P}
end

debug(x) = println(x)
