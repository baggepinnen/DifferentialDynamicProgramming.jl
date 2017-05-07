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
        ABd = expm([fx[:,:,ii]*h  fu[:,:,ii]*h; zeros(nu, D + nu)])# ZoH sampling
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


alpha        = logspace(0,-3,11)
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
alpha= logspace(0.2,-3,6)
verbosity=3
tol_fun = 1e-7
max_iter=1000


import Base: length
EmptyMat3 = Array(Float64,0,0,0)
EmptyMat2 = Array(Float64,0,0)
emptyMat3(P) = Array(P,0,0,0)
emptyMat2(P) = Array(P,0,0)
type Trace
    iter::Int64
    λ::Float64
    dλ::Float64
    cost::Float64
    alpha::Float64
    grad_norm::Float64
    improvement::Float64
    reduce_ratio::Float64
    time_derivs::Float64
    time_forward::Float64
    time_backward::Float64
    Trace() = new(0,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
end

type GaussianDist{P}
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
type GaussianTrajDist{P}
    policy::GaussianDist{P}
    dynamics::GaussianDist{P}
end

debug(x) = println(x)
function forward_pass(traj_new, x0,u,x,alpha,f,fT,lims,diff)
    n         = size(x0,1)
    m,N       = size(u)
    xnew      = Array(eltype(x0),n,N)
    xnew[:,1] = x0
    unew      = copy(u)
    cnew      = zeros(N)
    debug("Entering forward_pass loop")
    for i = 1:N
        if !isempty(traj_new)
            unew[:,i] .+= traj_new.μu[:,i]*alpha
        end
        if !isempty(traj_new.fx)
            dx = diff(xnew[:,i], x[:,i])
            unew[:,i] .+= traj_new.fx[:,:,i]*dx
        end
        if !isempty(lims)
            unew[:,i] = clamp.(unew[:,i],lims[:,1], lims[:,2])
        end
        xnewi, cnew[i]  = f(xnew[:,i], unew[:,i], i)
        if i < N
            xnew[:,i+1] = xnewi
        end
    end
    # cnew[N+1] = fT(xnew[:,N+1])
    sigmanew = zeros(m+n,m+n,N) # TODO: this function should calculate the covariance matrix as well
    return xnew,unew,cnew,sigmanew
end



# --- initial sizes and controls
n   = size(x0, 1)          # dimension of state vector
m   = size(u0, 1)          # dimension of control vector
N   = size(u0, 2)          # number of state transitions
u   = u0                   # initial control sequence
η   = 0.
traj_new  = GaussianDist(Float64)
traj_prev = GaussianDist(Float64)

# --- initialize trace data structure
trace = [Trace() for i in 1:min( max_iter+1,1e6)]
trace[1].iter = 1
trace[1].λ = λ
trace[1].dλ = dλ



# --- initial trajectory
debug("Setting up initial trajectory")
if size(x0,2) == 1 # only initial state provided
    diverge = true
    for alphai =  alpha
        debug("# test different backtracing parameters alphai and break loop when first succeeds")
        x,un,cost,_ = forward_pass(traj_new,x0[:,1],alphai*u,[],1,f,fT, lims,diff_fun)
        debug("# simplistic divergence test")
        if all(abs.(vec(x)) .< 1e8)
            u = un
            diverge = false
            break
        end
    end
elseif size(x0,2) == N
    debug("# pre-rolled initial forward pass, initial traj provided, e.g. from demonstration")
    x        = x0
    diverge  = false
    isempty(cost) && error("Initial trajectory supplied, initial cost must also be supplied")
else
    error("pre-rolled initial trajectory must be of correct length (size(x0,2) == N)")
end



# constants, timers, counters
flg_change = true
dcost      = 0.
expected   = 0.
div        = 0.
print_head = 10 # print headings every print_head lines
last_head  = print_head
g_norm     = Vector{Float64}()
t_start    = time()
verbosity > 0 && @printf("\n---------- begin iLQG ----------\n")

iter = accepted_iter = 1


trace[iter].iter = iter
back_pass_done   = false
l                = Matrix{Float64}()
dV               = Vector{Float64}()
reduce_ratio     = 0.
traj_prev        = traj_new
# ====== STEP 1: differentiate dynamics and cost along new trajectory
if flg_change
    tic()
    fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu = df(x, u)
    trace[iter].time_derivs = toq()
    flg_change   = false
end





macro setupQTIC()
    quote
        m   = size(u,1)
        n,N = size(fx,1,3)

        cx  = reshape(cx, (n, N))
        cu  = reshape(cu, (m, N))
        cxx = reshape(cxx, (n, n))
        cxu = reshape(cxu, (n, m))
        cuu = reshape(cuu, (m, m))

        k   = zeros(m,N)
        K   = zeros(m,n,N)
        Vx  = zeros(n,N)
        Vxx = zeros(n,n,N)
        dV  = [0., 0.] # TODO: WTF is dV?

        Vx[:,N]    = cx[:,N]
        Vxx[:,:,N] = cxx
        Quu        = Array(T,m,m,N)
        diverge    = 0
    end |> esc
end

macro end_backward_pass()
    quote
        if isempty(lims) || lims[1,1] > lims[1,2]
            debug("#  no control limits: Cholesky decomposition, check for non-PD")
            try
                R = chol(Hermitian(QuuF))
            catch
                diverge  = i
                return diverge, GaussianDist(N,n,m,K,[],Quu,x,u), Vx, Vxx
            end

            debug("#  find control law")
            kK  = -R\(R'\[Qu Qux_reg])
            k_i = kK[:,1]
            K_i = kK[:,2:n+1]
        else
            debug("#  solve Quadratic Program")
            lower = lims[:,1]-u[:,i]
            upper = lims[:,2]-u[:,i]

            k_i,result,R,free = boxQP(QuuF,Qu,lower,upper,k[:,min(i+1,N-1)])
            if result < 1
                diverge  = i
                return diverge, Vx, Vxx, k, K, dV
            end
            K_i  = zeros(m,n)
            if any(free)
                Lfree         = -R\(R'\Qux_reg[free,:])
                K_i[free,:]   = Lfree
            end
        end
        debug("#  update cost-to-go approximation")
        # Qxx         = Hermitian(Qxx)
        dV         = dV + [k_i'Qu; .5*k_i'Quu[:,:,i]*k_i]
        Vx[:,i]    = Qx + K_i'Quu[:,:,i]*k_i + K_i'Qu + Qux'k_i
        Vxx[:,:,i] = Qxx + K_i'Quu[:,:,i]*K_i + K_i'Qux + Qux'K_i
        Vxx[:,:,i] = .5*(Vxx[:,:,i] + Vxx[:,:,i]')

        debug("# save controls/gains")
        k[:,i]   = k_i
        K[:,:,i] = K_i
    end |> esc
end


function back_pass{T}(cx,cu,cxx::AbstractArray{T,2},cxu,cuu,fx::AbstractArray{T,3},fu,λ,regType,lims,x,u) # quadratic timeinvariant cost, linear time variant dynamics
    @setupQTIC
    for i = N-1:-1:1
        Qu         = cu[:,i] + fu[:,:,i]'Vx[:,i+1]
        Qx         = cx[:,i] + fx[:,:,i]'Vx[:,i+1]
        Qux        = cxu' + fu[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        Quu[:,:,i] = cuu + fu[:,:,i]'Vxx[:,:,i+1]*fu[:,:,i]
        Qxx        = cxx + fx[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        Vxx_reg    = Vxx[:,:,i+1] + (regType == 2 ? λ*eye(n) : 0)
        Qux_reg    = cxu' + fu[:,:,i]'Vxx_reg*fx[:,:,i]
        QuuF       = cuu + fu[:,:,i]'Vxx_reg*fu[:,:,i] + (regType == 1 ? λ*eye(m) : 0)

        @end_backward_pass
    end

    return diverge, GaussianDist(N,n,m,K,EmptyMat3,Quu,x,k), Vx, Vxx,dV
end

linearsys = isempty(fxx) && isempty(fxu) && isempty(fuu);

diverge, traj_new,Vx, Vxx,dV = if linearsys
    back_pass(cx,cu,cxx,cxu,cuu,fx,fu,λ, regType, lims,x,u)
else
    back_pass(cx,cu,cxx,cxu,cuu,fx,fu,fxx,fxu,fuu,λ, regType, lims,x,u)
end





k, K = traj_new.μu, traj_new.fx
#  check for termination due to small gradient
g_norm = mean(maximum(abs.(k) ./ (abs.(u)+1),1))
trace[iter].grad_norm = g_norm
if g_norm <  tol_grad && λ < 1e-5 && satisfied
    verbosity > 0 && @printf("\nSUCCESS: gradient norm < tol_grad\n")
    break
end


xnew,unew,costnew,sigmanew = Matrix{Float64}(),Matrix{Float64}(),Vector{Float64}(),Matrix{Float64}()


for alphai = alpha
    xnew,unew,costnew,sigmanew = forward_pass(traj_new, x0[:,1] ,u, x,alphai,f,fT, lims, diff_fun)
    dcost    = sum(cost) - sum(costnew)
    expected = -alphai*(dV[1] + alphai*dV[2])
    reduce_ratio = if expected > 0
        dcost/expected
    else
        warn("negative expected reduction: should not occur")
        sign(dcost)
    end
    if reduce_ratio > reduce_ratio_min
        fwd_pass_done = true
        break
    end
end
