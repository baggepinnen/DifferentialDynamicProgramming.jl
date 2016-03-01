using Sundials
using Control
using ReverseDiffSource


T     = 100
N     = T+1
g     = 9.82
l     = 0.35
h     = 0.05
x0    = [3.0, 0, 0, 0]
lims  = 20.0*[-1 1] # control limits, e.g. ones(m,1)*[-1 1]*.6
goal  = [pi,0,0,0]
A     = [0 1 0 0;
         g/l 0 0 0;
         0 0 0 1;
         0 0 0 0]
B     = [0, -1/l, 0, 1]
C     = eye(4)
D     = 4

sys   = ss(A,B,C,zeros(4))
Q     = h*diagm([10,5,1,10])
R     = h*1

function fsys(t,x,xd)
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

function dfsys(x,u, t)
    x + h*[x[2]; -g/l*sin(x[1])+u/l*cos(x[1]); x[4]; u]
end

# df_ex = :(x[1:4] + h*[x[2], -g/l * sin(x[1]) + x[5]/l * cos(x[1]), x[4], x[5]])
# ddf_ex = rdiff(df_ex, x=[x0; 0.0], order = 2)
# @eval ddf = $ddf_ex

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
    f = dfsys(x,u,i)
    c = cost_quadratic(x,u)
    return f,c
end

function lin_dyn_fT(x)
    cost_quadratic(x,0.0)
end


function lin_dyn_df(x,u,i::UnitRange)
    u[isnan(u)] = 0
    I = length(i)
    D = size(x,1)
    fx = Array{Float64}(D,D,I)
    fu = Array{Float64}(D,1,I)
    cx,cu,cxu = dcost_quadratic(x,u)
    cxx = Q
    cuu = [R]
    for ii = i
        fx[:,:,ii] = [0 1 0 0;
                      -g/l*cos(x[1,ii])-u[ii]/l*sin(x[1,ii]) 0 0 0;
                      0 0 0 1;
                      0 0 0 0]

        fu[:,:,ii] = [0, cos(x[1,ii])/l, 0, 1]
    end
    fxx=fxu=fuu = []
    return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
end

function lin_dyn_df(x,u,i)
    u[isnan(u)] = 0
    D = size(x,1)
    fx = Array{Float64}(D,D)
    fu = Array{Float64}(D,1)
    cx,cu,cxu = dcost_quadratic(x,u)
    cxx = Q
    cuu = R
    fx = [0 1 0 0;
          -g/l*cos(x[1])-u/l*sin(x[1]) 0 0 0;
          0 0 0 1;
          0 0 0 0]

    fu = [0, cos(x[1])/l, 0, 1]
    fxx=fxu=fuu = []
    return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
end





# Control.lqr(sys,Q,R)
L = R\B'*care(A, B, Q, R)
L += 0.3L.*randn(1,4)
R     = h*1
dof   = 1


# x = Sundials.cvode(f, x0, collect(0:h:5),1.0)
# plot(res)
# T = size(x,1)





function simulate_pendcart()

    x = zeros(4,N)
    u = zeros(1,T)
    x[:,1] = [2.5,0,0,0]
    u[1] = 0

    for t = 2:T
        dx     = copy(x[:,t-1])
        dx[1] -= pi
        u[t]   = saturate(-(L*dx)[1])
        x[:,t] = dfsys(x[:,t-1],u[t],0)
    end
    dx     = copy(x[:,T])
    dx[1] -= pi
    uT   = saturate(-(L*dx)[1])
    x[:,T+1] = dfsys(x[:,T],uT,0)
    cost = cost_quadratic(x,u)
    #     plotsub([x' u])
    #     show()
    return x, u, cost
end

function saturate(u)
    u = max(u,lims[1])
    u = min(u,lims[2])
end

x0, u0, cost = simulate_pendcart()
