# DifferentialDynamicProgramming

[![Build Status](https://travis-ci.org/baggepinnen/DifferentialDynamicProgramming.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/DifferentialDynamicProgramming.jl)

[![Coverage Status](https://coveralls.io/repos/github/baggepinnen/DifferentialDynamicProgramming.jl/badge.png?branch=master)](https://coveralls.io/github/baggepinnen/DifferentialDynamicProgramming.jl?branch=master)

This code consists of a port and extensions of a MATLAB library provided by the autors of
>BIBTeX:
>@INPROCEEDINGS{
>  author={Tassa, Y. and Mansard, N. and Todorov, E.},
>  booktitle={Robotics and Automation (ICRA), 2014 IEEE International Conference on},
>  title={Control-Limited Differential Dynamic Programming},
>  year={2014}, month={May}, doi={10.1109/ICRA.2014.6907001}}
>}

>http://www.cs.washington.edu/people/postdocs/tassa/

All users of this package for academic purposes are encouraged to cite the original article.



## Demo functions
The following demo functions are provided
`demo_linear()` To run the iLQG DDP algorithm on a simple linear problem
`demoQP` To solve a demo quadratic program

## Usage
See demo file `demo_linear.jl` for a usage example.

```julia
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

# Specify dynamics functions
function lin_dyn_df(x,u,Q,R)
    u[isnan(u)] = 0
    cx  = Q*x
    cu  = R*u
    fxx=fxu=fuu = []
    return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
end
function lin_dyn_f(x,u,A,B,Q,R)
    u[isnan(u)] = 0
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
@time x, u, L, Vx, Vxx, cost, otrace = iLQG(f,fT,df, x0, u0, lims=lims, plotFn= x -> 0 );
```


