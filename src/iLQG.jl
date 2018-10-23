import Base: length
EmptyMat3 = Array{Float64}(undef, 0,0,0)
EmptyMat2 = Array{Float64}(undef, 0,0)
emptyMat3(P) = Array{P}(undef, 0,0,0)
emptyMat2(P) = Array{P}(undef, 0,0)
mutable struct Trace
    iter::Int64
    λ::Float64
    dλ::Float64
    cost::Float64
    α::Float64
    grad_norm::Float64
    improvement::Float64
    reduce_ratio::Float64
    time_derivs::Float64
    time_forward::Float64
    time_backward::Float64
    divergence::Float64
    η::Float64
    Trace() = new(0,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
end

(t::MVHistory)(args...) = increment!(t, args...)

"""
    `GaussianPolicy{P}`

# Fileds:
```
T::Int          # number of time steps
n::Int          # State dimension
m::Int          # Number of control inputs
K::Array{P,3}   # Time-varying feedback gain ∈ R(n,m,T)
k::Array{P,2}   # Open loop control signal  ∈ R(m,T)
Σ::Array{P,3}   # Time-varying controller covariance  ∈ R(m,m,T)
Σi::Array{P,3}  # The inverses of Σ
```
"""
mutable struct GaussianPolicy{P}
    T::Int
    n::Int
    m::Int
    K::Array{P,3}
    k::Array{P,2}
    Σ::Array{P,3}
    Σi::Array{P,3}
end

eye(P,n) = Matrix{P}(I,n,n)
GaussianPolicy(P) = GaussianPolicy(0,0,0,emptyMat3(P),emptyMat2(P),emptyMat3(P),emptyMat3(P))
GaussianPolicy(P,T,n,m) = GaussianPolicy(T,n,m,zeros(P,m,n,T),zeros(P,m,T),cat([eye(P,m) for t=1:T]..., dims=3),cat([eye(P,m) for t=1:T]..., dims=3))
Base.isempty(gp::GaussianPolicy) = gp.T == gp.n == gp.m == 0
Base.length(gp::GaussianPolicy) = gp.T

include("klutils.jl")

"""
iLQG - solve the deterministic finite-horizon optimal control problem.

minimize sum_i cost(x[:,i],u[:,i]) + cost(x[:,end])
s.t.  x[:,i+1] = f(x[:,i],u[:,i])

Inputs
======
`f, costfun, df`

1) step:
`xnew = f(x,u,i)` is called during the forward pass.
Here the state x and control u are vectors: size(x)==(n,),
size(u)==(m,). The time index `i` is a scalar.


2) cost:
`cost = costfun(x,u)` is called in the forward pass to compute
the cost per time-step along the trajectory `x,u`.

3) derivatives:
`fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu = df(x,u)` computes the
derivatives along a trajectory. In this case size(x)==(n, N) where N
is the trajectory length. size(u)==(m, N). The time indexes are I=(1:N).
Dimensions match the variable names e.g. size(fxu)==(n, n, m, N)
If cost function or system is time invariant, the dimension of the corresponding
derivatives can be reduced by dropping the time dimension

`x0` - The initial state from which to solve the control problem.
Should be a column vector. If a pre-rolled trajectory is available
then size(x0)==(n, N) can be provided and cost set accordingly.

`u0` - The initial control sequence. A matrix of size(u0)==(m, N)
where m is the dimension of the control and N is the number of state
transitions.

Outputs
=======
`x` - the optimal state trajectory found by the algorithm.
size(x)==(n, N)

`u` - the optimal open-loop control sequence.
size(u)==(m, N)

`traj_new` - A new `GaussianPolicy` object containing feedforward control trajectory and feedback-gains, these gains multiply the
deviation of a simulated trajectory from the nominal trajectory x. See `?GaussianPolicy` for more help.

`Vx` - the gradient of the cost-to-go. size(Vx)==(n, N)

`Vxx` - the Hessian of the cost-to-go. size(Vxx)==(n, n N)

`cost` - the costs along the trajectory. size(cost)==(1, N)
the cost-to-go is V = fliplr(cumsum(fliplr(cost)))

`trace` - a trace of various convergence-related values. One row for each
iteration, the columns of trace are
`[iter λ α g_norm Δcost z sum(cost) dλ]`
see below for details.

# Keyword arguments
`lims`,           [],            control limits\n
`α`,              logspace(0,-3,11), backtracking coefficients\n
`tol_fun`,         1e-7,          reduction exit criterion\n
`tol_grad`,        1e-4,          gradient exit criterion\n
`max_iter`,        500,           maximum iterations\n
`λ`,         1,             initial value for λ\n
`dλ`,        1,             initial value for dλ\n
`λfactor`,   1.6,           λ scaling factor\n
`λmax`,      1e10,          λ maximum value\n
`λmin`,      1e-6,          below this value λ = 0\n
`regType`,        1,             regularization type 1: q_uu+λ*I 2: V_xx+λ*I\n
`reduce_ratio_min`,           0,             minimal accepted reduction ratio\n
`diff_fun`,         -,             user-defined diff for sub-space optimization\n
`plot`,           1,             0: no  k>0: every k iters k<0: every k iters, with derivs window\n
`verbosity`,      2,             0: no  1: final 2: iter 3: iter, detailed\n
`plot_fun`,         x->0,          user-defined graphics callback\n
`cost`,           [],            initial cost for pre-rolled trajectory

This code consists of a port and extension of a MATLAB library provided by the autors of
`   INPROCEEDINGS{author={Tassa, Y. and Mansard, N. and Todorov, E.},
booktitle={Robotics and Automation (ICRA), 2014 IEEE International Conference on},
title={Control-Limited Differential Dynamic Programming},
year={2014}, month={May}, doi={10.1109/ICRA.2014.6907001}}`
"""
function iLQG(f,costfun,df, x0, u0;
    lims             = [],
    α                = exp10.(range(0, stop=-3, length=11)),
    tol_fun          = 1e-7,
    tol_grad         = 1e-4,
    max_iter         = 500,
    λ                = 1.,
    dλ               = 1.,
    λfactor          = 1.6,
    λmax             = 1e10,
    λmin             = 1e-6,
    regType          = 1,
    reduce_ratio_min = 0,
    diff_fun         = -,
    plot             = 1,
    verbosity        = 2,
    plot_fun         = x->0,
    cost             = [],
    traj_prev        = 0
    )
    debug("Entering iLQG")
    local fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu,xnew,unew,costnew,g_norm,Vx,Vxx,dV,αi
    # --- initial sizes and controls
    n   = size(x0, 1)          # dimension of state vector
    m   = size(u0, 1)          # dimension of control vector
    N   = size(u0, 2)          # number of state transitions
    u   = u0                   # initial control sequence
    traj_new  = GaussianPolicy(Float64)
    # traj_prev = GaussianDist(Float64)

    # --- initialize trace data structure
    trace = MVHistory()
    trace(:λ, 0, λ)
    trace(:dλ, 0, dλ)

    # --- initial trajectory
    debug("Setting up initial trajectory")
    if size(x0,2) == 1 # only initial state provided
        diverge = true
        for outer αi ∈ α
            debug("# test different backtracing parameters α and break loop when first succeeds")
            x,un,cost, = forward_pass(traj_new,x0[:,1],αi*u,[],1,f,costfun, lims,diff_fun)
            debug("# simplistic divergence test")
            if all(abs.(x) .< 1e8)
                u = un
                diverge = false
                break
            end
        end
    elseif size(x0,2) == N
        debug("# pre-rolled initial forward pass, initial traj provided")
        x        = x0
        diverge  = false
        isempty(cost) && error("Initial trajectory supplied, initial cost must also be supplied")
    else
        error("pre-rolled initial trajectory must be of correct length (size(x0,2) == N)")
    end

    trace(:cost, 0, sum(cost))
    #     plot_fun(x) # user plotting

    if diverge
        if verbosity > 0
            @printf("\nEXIT: Initial control sequence caused divergence\n")
        end
        return
    end

    # constants, timers, counters
    flg_change         = true
    Δcost              = 0.
    expected_reduction = 0.
    print_head         = 10 # print headings every print_head lines
    last_head          = print_head
    t_start            = time()
    verbosity > 0 && @printf("\n---------- begin iLQG ----------\n")
    satisfied          = true

    iter = accepted_iter = 1
    while accepted_iter <= max_iter
        reduce_ratio     = 0.
        # ====== STEP 1: differentiate dynamics and cost along new trajectory
        if flg_change
            _t = @elapsed fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu = df(x, u)
            trace(:time_derivs, iter, _t)
            flg_change   = false
        end
        # Determine what kind of system we are dealing with
        linearsys = isempty(fxx) && isempty(fxu) && isempty(fuu); debug("linear system: $linearsys")

        # ====== STEP 2: backward pass, compute optimal control law and cost-to-go
        back_pass_done = false
        while !back_pass_done
            _t = @elapsed diverge, traj_new,Vx, Vxx,dV = if linearsys
                back_pass(cx,cu,cxx,cxu,cuu,fx,fu,λ, regType, lims,x,u)
            else
                back_pass(cx,cu,cxx,cxu,cuu,fx,fu,fxx,fxu,fuu,λ, regType, lims,x,u)
            end
            increment!(trace, :time_backward, iter, _t)
            iter == 1 && (traj_prev = traj_new) # TODO: set k μu to zero fir traj_prev

            if diverge > 0
                verbosity > 2 && @printf("Cholesky failed at timestep %d.\n",diverge)
                dλ,λ = max(dλ*λfactor, λfactor), max(λ*dλ, λmin)
                if λ >  λmax; break; end
                continue
            end
            back_pass_done = true
        end


        k, K = traj_new.k, traj_new.K
        #  check for termination due to small gradient
        g_norm = mean(maximum(abs.(k) ./ (abs.(u) .+ 1), dims=1))
        trace(:grad_norm, iter, g_norm)
        if g_norm <  tol_grad && λ < 1e-5 && satisfied
            verbosity > 0 && @printf("\nSUCCESS: gradient norm < tol_grad\n")
            break
        end

        # ====== STEP 3: line-search to find new control sequence, trajectory, cost
        fwd_pass_done  = false
        if back_pass_done
            debug("#  serial backtracking line-search")
            @elapsed(for outer αi = α
                xnew,unew,costnew = forward_pass(traj_new, x0[:,1] ,u, x,αi,f,costfun, lims, diff_fun)
                Δcost    = sum(cost) - sum(costnew)
                expected_reduction = -αi*(dV[1] + αi*dV[2])
                reduce_ratio = if expected_reduction > 0
                    Δcost/expected_reduction
                else
                    @warn("negative expected reduction: should not occur")
                    sign(Δcost)
                end
                if reduce_ratio > reduce_ratio_min
                    fwd_pass_done = true
                    break
                end
            end) |> x -> trace(:time_forward, iter, x)

        end

        # ====== STEP 4: accept step (or not), print status

        #  print headings
        if verbosity > 1 && last_head == print_head
            last_head = 0
            @printf("%-12s", "iteration     cost    reduction     expected    gradient    log10(λ)    η    divergence\n")
        end

        if fwd_pass_done && satisfied # TODO: I added satisfied here, verify if this is reasonable
            if verbosity > 1
                @printf("%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f\n",
                iter, sum(cost), Δcost, expected_reduction, g_norm, log10(λ))
                last_head += 1
            end
            dλ = min(dλ / λfactor, 1/ λfactor)
            λ *= dλ
            #  accept changes
            x,u,cost  = copy(xnew),copy(unew),copy(costnew)
            traj_new.k = copy(u)
            flg_change = true
            plot_fun(x)
            if Δcost < tol_fun
                verbosity > 0 &&  @printf("\nSUCCESS: cost change < tol_fun\n")
                break
            end
            accepted_iter += 1
        else #  no cost improvement
            αi =  NaN
            dλ,λ  = max(dλ * λfactor,  λfactor), max(λ * dλ,  λmin)#  increase λ
            if verbosity > 1
                @printf("%-12d%-12s%-12.3g%-12.3g%-12.3g%-12.1f\n",
                iter,"NO STEP", Δcost, expected_reduction, g_norm, log10(λ))
                last_head = last_head+1
            end
            if λ > λmax #  terminate ?
                verbosity > 0 && @printf("\nEXIT: λ > λmax\n")
                break
            end
        end
        #  update trace
        trace(:λ, iter, λ)
        trace(:dλ, iter, dλ)
        trace(:α, iter, αi)
        trace(:improvement, iter, Δcost)
        trace(:cost, iter, sum(cost))
        trace(:reduce_ratio, iter, reduce_ratio)
        iter += 1
    end

    iter ==  max_iter &&  verbosity > 0 && @printf("\nEXIT: Maximum iterations reached.\n")
    iter == 1 && error("Failure: no iterations completed, something is wrong. Try enabling the debug flag in DifferentialDynamicProgramming.jl for verbose printing.")


    verbosity > 0 && print_timing(trace,iter,t_start,cost,g_norm,λ)

    return x, u, traj_new, Vx, Vxx, cost, trace
end

function print_timing(trace,iter,t_start,cost,g_norm,λ)
    diff_t  = get(trace, :time_derivs)[2]
    diff_t  = sum(diff_t[.!isnan.(diff_t)])
    back_t  = get(trace, :time_backward)[2]
    back_t  = sum(back_t[.!isnan.(back_t)])
    fwd_t   = get(trace, :time_forward)[2]
    fwd_t   = sum(fwd_t[.!isnan.(fwd_t)])
    total_t = time()-t_start
    info = 100/total_t*[diff_t, back_t, fwd_t, (total_t-diff_t-back_t-fwd_t)]
    try
        @printf("\n iterations:   %-3d\n
        final cost:   %-12.7g\n
        final grad:   %-12.7g\n
        final λ: %-12.7e\n
        time / iter:  %-5.0f ms\n
        total time:   %-5.2f seconds, of which\n
        derivs:     %-4.1f%%\n
        back pass:  %-4.1f%%\n
        fwd pass:   %-4.1f%%\n
        other:      %-4.1f%% (graphics etc.)\n =========== end iLQG ===========\n",iter,sum(cost),g_norm,λ,1e3*total_t/iter,total_t,info[1],info[2],info[3],info[4])
    catch
        @show g_norm
    end
end
