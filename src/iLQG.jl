mutable struct Trace
    iter::Int64
    lambda::Float64
    dlambda::Float64
    cost::Float64
    alpha::Float64
    grad_norm::Float64
    improvement::Float64
    reduc_ratio::Float64
    time_derivs::Float64
    time_forward::Float64
    time_backward::Float64
    Trace() = new(0,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
end

# type DiffDynCost{T}
#     fx::VecOrMat{T}
#     fu::Vector{T}
#     fxx::Matrix{T}
#     fxu::Vector{T}
#     fuu::Vector{T}
#     cx::Vector{T}
#     cu::Vector{T}
#     cxx::Vector{T}
#     cxu::Vector{T}
#     cuu::Vector{T}
# end



"""
iLQG - solve the deterministic finite-horizon optimal control problem.

minimize sum_i cost(x[:,i],u[:,i]) + cost(x[:,end])
s.t.  x[:,i+1] = f(x[:,i],u[:,i])

Inputs
======
`f, fT, df`

1) step:
`xnew,cost = f(x,u,i)` is called during the forward pass.
Here the state x and control u are vectors: size(x)==(n,1),
size(u)==(m,1). The cost and time index `i` are scalars.


2) final:
`cost = fT(x)` is called at the end the forward pass to compute
the final cost.

3) derivatives:
`fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu = df(x,u,I)` computes the
derivatives along a trajectory. In this case size(x)==(n, N+1) where N
is the trajectory length. size(u)==(m, N+1) with NaNs in the last column
to indicate final-cost. The time indexes are I=(1:N).
Dimensions match the variable names e.g. size(fxu)==(n, n, m, N+1)
note that the last temporal element N+1 is ignored for all tensors
except cx and cxx, the final-cost derivatives.
If cost function or system is time invariant, the dimension of the corresponding
derivatives can be reduced by dropping the time dimension

`x0` - The initial state from which to solve the control problem.
Should be a column vector. If a pre-rolled trajectory is available
then size(x0)==(n, N+1) can be provided and cost set accordingly.

`u0` - The initial control sequence. A matrix of size(u0)==(m, N)
where m is the dimension of the control and N is the number of state
transitions.

Outputs
=======
`x` - the optimal state trajectory found by the algorithm.
size(x)==(n, N+1)

`u` - the optimal open-loop control sequence.
size(u)==(m, N)

`L` - the optimal closed loop control gains. These gains multiply the
deviation of a simulated trajectory from the nominal trajectory x.
size(L)==(m, n N)

`Vx` - the gradient of the cost-to-go. size(Vx)==(n, N+1)

`Vxx` - the Hessian of the cost-to-go. size(Vxx)==(n, n N+1)

`cost` - the costs along the trajectory. size(cost)==(1, N+1)
the cost-to-go is V = fliplr(cumsum(fliplr(cost)))

`lambda` - the final value of the regularization parameter

`trace` - a trace of various convergence-related values. One row for each
iteration, the columns of trace are
`[iter lambda alpha g_norm dcost z sum(cost) dlambda]`
see below for details.

`timing` - timing information
---------------------- user-adjustable parameters ------------------------
defaults

`lims`,           [],            control limits
`Alpha`,          logspace(0,-3,11), backtracking coefficients
`tolFun`,         1e-7,          reduction exit criterion
`tolGrad`,        1e-4,          gradient exit criterion
`maxIter`,        500,           maximum iterations
`lambda`,         1,             initial value for lambda
`dlambda`,        1,             initial value for dlambda
`lambdaFactor`,   1.6,           lambda scaling factor
`lambdaMax`,      1e10,          lambda maximum value
`lambdaMin`,      1e-6,          below this value lambda = 0
`regType`,        1,             regularization type 1: q_uu+lambda*I 2: V_xx+lambda*I
`zMin`,           0,             minimal accepted reduction ratio
`diffFn`,         -,             user-defined diff for sub-space optimization
`plot`,           1,             0: no  k>0: every k iters k<0: every k iters, with derivs window
`verbosity`,      2,             0: no  1: final 2: iter 3: iter, detailed
`plotFn`,         x->0,          user-defined graphics callback
`cost`,           [],            initial cost for pre-rolled trajectory

This code consists of a port and extension of a MATLAB library provided by the autors of
`   INPROCEEDINGS{author={Tassa, Y. and Mansard, N. and Todorov, E.},
booktitle={Robotics and Automation (ICRA), 2014 IEEE International Conference on},
title={Control-Limited Differential Dynamic Programming},
year={2014}, month={May}, doi={10.1109/ICRA.2014.6907001}}`
"""
function iLQG(f,fT,df, x0, u0;
    lims=           [],
    Alpha=          logspace(0,-3,11),
    tolFun=         1e-7,
    tolGrad=        1e-4,
    maxIter=        500,
    lambda=         1,
    dlambda=        1,
    lambdaFactor=   1.6,
    lambdaMax=      1e10,
    lambdaMin=      1e-6,
    regType=        1,
    zMin=           0,
    diffFn=         -,
    plot=           1,
    verbosity=      2,
    plotFn=         x->0,
    cost=           [],
    )
    debug("Entering iLQG")


    # --- initial sizes and controls
    n   = size(x0, 1)          # dimension of state vector
    m   = size(u0, 1)          # dimension of control vector
    N   = size(u0, 2)          # number of state transitions
    u   = u0                   # initial control sequence

    # --- initialize trace data structure
    trace = [Trace() for i in 1:min( maxIter+1,1e6)]
    trace[1].iter = 1
    trace[1].lambda = lambda
    trace[1].dlambda = dlambda

    # --- initial trajectory
    debug("Setting up initial trajectory")
    if size(x0,2) == 1 # only initial state provided
        diverge = true
        for alpha =  Alpha
            debug("# test different backtracing parameters alpha and break loop when first succeeds")
            (x,un,cost)  = forward_pass(x0[:,1],alpha*u,[],[],[],1,f,fT, lims,[])
            debug("# simplistic divergence test")
            if all(abs.(x) .< 1e8)
                u = un
                diverge = false
                break
            end
        end
    elseif size(x0,2) == N+1
        debug("# pre-rolled initial forward pass, initial traj provided, e.g. from demonstration")
        x        = x0
        diverge  = false
        isempty(cost) && error("Initial trajectory supplied, initial cost must also be supplied")
    else
        error("pre-rolled initial trajectory must be of correct length (size(x0,2) == N+1)")
    end

    trace[1].cost = sum(cost)

    # user plotting
    #     plotFn(x)

    if diverge
        Vx=Vxx   = NaN
        L        = zeros(m,n,N)
        cost     = []
        trace    = trace[1]
        if verbosity > 0
            @printf("\nEXIT: Initial control sequence caused divergence\n")
        end
        return
    end

    # constants, timers, counters
    flgChange   = true
    dcost       = 0.
    z           = 0
    expected    = 0.
    print_head  = 10 # print headings every print_head lines
    last_head   = print_head
    g_norm      = Vector{Float64}()
    t_start     = time()
    verbosity > 0 && @printf("\n---------- begin iLQG ----------\n")


    iter = 1
    while iter <= maxIter
        trace[iter].iter = iter
        # ====== STEP 1: differentiate dynamics and cost along new trajectory
        if flgChange
            tic()
            fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu = df(x, u , 1:N)
            trace[iter].time_derivs = toq()
            flgChange   = false
        end

        # Determine what kind of system we are dealing with
        linearsys = isempty(fxx) && isempty(fxu) && isempty(fuu)
        debug("linear system: $linearsys")

        # ====== STEP 2: backward pass, compute optimal control law and cost-to-go
        backPassDone   = false
        l = Matrix{Float64}(0,0)
        dV = Vector{Float64}(0)
        while !backPassDone

            tic()
            if linearsys
                diverge, Vx, Vxx, l, L, dV = back_pass(cx,cu,cxx,cxu,cuu,fx,fu,lambda, regType, lims,u)
            else
                diverge, Vx, Vxx, l, L, dV = back_pass(cx,cu,cxx,cxu,cuu,fx,fu,fxx,fxu,fuu,lambda, regType, lims,u)
            end
            trace[iter].time_backward = toq()

            if diverge > 0
                if verbosity > 2
                    @printf("Cholesky failed at timestep %d.\n",diverge)
                end
                dlambda   = max(dlambda *  lambdaFactor,  lambdaFactor)
                lambda    = max(lambda * dlambda,  lambdaMin)
                if lambda >  lambdaMax; break; end
                continue
            end
            backPassDone = true
        end

        #  check for termination due to small gradient
        g_norm = mean(maximum(abs.(l) ./ (abs.(u)+1),1))
        trace[iter].grad_norm = g_norm
        if g_norm <  tolGrad && lambda < 1e-5
            # dlambda   = min(dlambda /  lambdaFactor, 1/ lambdaFactor)
            # lambda    = lambda * dlambda * (lambda >  lambdaMin)
            if verbosity > 0
                @printf("\nSUCCESS: gradient norm < tolGrad\n")
            end
            break
        end

        # ====== STEP 3: line-search to find new control sequence, trajectory, cost
        fwdPassDone  = false
        xnew = Matrix{Float64}
        unew = Matrix{Float64}
        costnew = Vector{Float64}
        if backPassDone
            tic()
            debug("#  serial backtracking line-search")
            for alpha =  Alpha
                xnew,unew,costnew   = forward_pass(x0[:,1] ,u+l*alpha, L, x,[],1,f,fT, lims, diffFn)
                dcost    = sum(cost) - sum(costnew)
                expected = -alpha*(dV[1] + alpha*dV[2])
                if expected > 0
                    z = dcost/expected
                else
                    z = sign(dcost)
                    warn("negative expected reduction: should not occur")
                end
                if z > zMin
                    fwdPassDone = true
                    break
                end
            end

            if !fwdPassDone
                alpha = NaN #  signals failure of forward pass
            end
            trace[iter].time_forward = toq()
        end

        # ====== STEP 4: accept step (or not), print status

        #  print headings
        if verbosity > 1 && last_head == print_head
            last_head = 0
            @printf("%-12s", "iteration     cost    reduction     expected    gradient    log10(lambda)")
            @printf("\n")
        end

        if fwdPassDone
            #  print status
            if verbosity > 1
                @printf("%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f\n",
                iter, sum(cost), dcost, expected, g_norm, log10(lambda))
                last_head += 1
            end

            #  decrease lambda
            dlambda   = min(dlambda /  lambdaFactor, 1/ lambdaFactor)
            lambda    = lambda * dlambda

            #  accept changes
            u              = unew
            x              = xnew
            cost           = costnew
            flgChange      = true
            plotFn(x)

            #  terminate ?
            if dcost < tolFun
                verbosity > 0 &&  @printf("\nSUCCESS: cost change < tolFun\n")
                break
            end
        else #  no cost improvement
            #  increase lambda
            dlambda  = max(dlambda * lambdaFactor,  lambdaFactor)
            lambda   = max(lambda * dlambda,  lambdaMin)

            #  print status
            if verbosity > 1
                @printf("%-12d%-12s%-12.3g%-12.3g%-12.3g%-12.1f\n",
                iter,"NO STEP", dcost, expected, g_norm, log10(lambda))
                last_head = last_head+1
            end

            #  terminate ?
            if lambda > lambdaMax
                verbosity > 0 && @printf("\nEXIT: lambda > lambdaMax\n")
                break
            end
        end
        #  update trace
        trace[iter].lambda      = lambda
        trace[iter].dlambda     = dlambda
        trace[iter].alpha       = alpha
        trace[iter].improvement = dcost
        trace[iter].cost        = sum(cost)
        trace[iter].reduc_ratio = z
        graphics( plot,x,u,cost,L,Vx,Vxx,fx,fxx,fu,fuu,trace[1:iter],0)
        iter += 1
    end

    #  save lambda/dlambda
    trace[iter].lambda      = lambda
    trace[iter].dlambda     = dlambda

    iter ==  maxIter &&  verbosity > 0 && @printf("\nEXIT: Maximum iterations reached.\n")


    if iter > 1
        diff_t = [trace[i].time_derivs for i in 1:iter]
        diff_t = sum(diff_t[.!isnan.(diff_t)])
        back_t = [trace[i].time_backward for i in 1:iter]
        back_t = sum(back_t[.!isnan.(back_t)])
        fwd_t = [trace[i].time_forward for i in 1:iter]
        fwd_t = sum(fwd_t[.!isnan.(fwd_t)])
        total_t = time()-t_start
        if verbosity > 0
            info = 100/total_t*[diff_t, back_t, fwd_t, (total_t-diff_t-back_t-fwd_t)]
            @printf("\n iterations:   %-3d\n
            final cost:   %-12.7g\n
            final grad:   %-12.7g\n
            final lambda: %-12.7e\n
            time / iter:  %-5.0f ms\n
            total time:   %-5.2f seconds, of which\n
            derivs:     %-4.1f%%\n
            back pass:  %-4.1f%%\n
            fwd pass:   %-4.1f%%\n
            other:      %-4.1f%% (graphics etc.)\n =========== end iLQG ===========\n",iter,sum(cost[:]),g_norm,lambda,1e3*total_t/iter,total_t,info[1],info[2],info[3],info[4])
        end
    else
        error("Failure: no iterations completed, something is wrong.")
    end

    return x, u, L, Vx, Vxx, cost, trace

end

function forward_pass(x0,u,L,x,du,Alpha,f,fT,lims,diff)

    n        = size(x0,1)
    m        = size(u,1)
    N        = size(u,2)

    xnew        = zeros(n,N+1)
    xnew[:,1]   = x0
    unew        = zeros(m,N)
    cnew        = zeros(N+1)
    debug("Entering forward_pass loop")
    for i = 1:N
        unew[:,i] = u[:,i]
        if !isempty(du)
            unew[:,i] = unew[:,i] + du[:,i]*Alpha
        end
        if !isempty(L)
            dx = diff(xnew[:,i], x[:,i])
            unew[:,i] = unew[:,i] + L[:,:,i]*dx
        end
        if !isempty(lims)
            unew[:,i] = min(lims[:,2], max(lims[:,1], unew[:,i]))
        end
        xnew[:,i+1], cnew[i]  = f(xnew[:,i], unew[:,i], i)
    end
    cnew[N+1] = fT(xnew[:,N+1])

    return xnew,unew,cnew
end

macro end_backward_pass()
    quote
        if isempty(lims) || lims[1,1] > lims[1,2]
            debug("#  no control limits: Cholesky decomposition, check for non-PD")
            try
                R = chol(Hermitian(QuuF))
            catch
                diverge  = i
                return diverge, Vx, Vxx, k, K, dV
            end

            debug("#  find control law")
            kK = -R\(R'\[Qu Qux_reg])
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
        dV          = dV + [k_i'Qu; .5*k_i'Quu*k_i]
        Vx[:,i]     = Qx  + K_i'Quu*k_i + K_i'Qu  + Qux'k_i
        Vxx[:,:,i]  = Qxx + K_i'Quu*K_i + K_i'Qux + Qux'K_i
        Vxx[:,:,i]  = .5*(Vxx[:,:,i] + Vxx[:,:,i]')

        debug("#  save controls/gains")
        k[:,i]      = k_i
        K[:,:,i]    = K_i
    end |> esc
end

macro setupQTIC()
    quote
        m     = size(u,1)
        n     = size(fx,1)
        N     = length(cx) ÷ n

        cx    = reshape(cx,  (n, N))
        cu    = reshape(cu,  (m, N-1))
        cxx   = reshape(cxx, (n, n))
        cxu   = reshape(cxu, (n, m))
        cuu   = reshape(cuu, (m, m))

        k     = zeros(m,N-1)
        K     = zeros(m,n,N-1)
        Vx    = zeros(n,N)
        Vxx   = zeros(n,n,N)
        dV    = [0., 0.]

        Vx[:,N]     = cx[:,N]
        Vxx[:,:,N]  = cxx
        diverge  = 0
    end |> esc
end

vectens(a,b) = permutedims(sum(a.*b,1), [3 2 1])

function back_pass{T}(cx,cu,cxx::AbstractArray{T,3},cxu,cuu,fx::AbstractArray{T,3},fu,fxx,fxu,fuu,lambda,regType,lims,u) where T# nonlinear time variant

    (m,N)  = size(u)
    n  = length(cx) ÷ N

    cx    = reshape(cx,  (n, N))
    cu    = reshape(cu,  (m, N))
    cxx   = reshape(cxx, (n, n, N))
    cxu   = reshape(cxu, (n, m, N))
    cuu   = reshape(cuu, (m, m, N))

    k     = zeros(m,N-1)
    K     = zeros(m,n,N-1)
    Vx    = zeros(n,N)
    Vxx   = zeros(n,n,N)
    dV    = [0., 0.]

    Vx[:,N]     = cx[:,N]
    Vxx[:,:,N]  = cxx[:,:,N]

    diverge  = 0
    for i = N-1:-1:1
        Qu  = cu[:,i]      + fu[:,:,i]'Vx[:,i+1]
        Qx  = cx[:,i]      + fx[:,:,i]'Vx[:,i+1]
        Qux = cxu[:,:,i]'  + fu[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        if !isempty(fxu)
            fxuVx = vectens(Vx[:,i+1],fxu[:,:,:,i])
            Qux   = Qux + fxuVx
        end

        Quu = cuu[:,:,i]   + fu[:,:,i]'Vxx[:,:,i+1]*fu[:,:,i]
        if !isempty(fuu)
            fuuVx = vectens(Vx[:,i+1],fuu[:,:,:,i])
            Quu   = Quu + fuuVx
        end

        Qxx = cxx[:,:,i]   + fx[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        if !isempty(fxx)
            Qxx = Qxx + vectens(Vx[:,i+1],fxx[:,:,:,i])
        end

        Vxx_reg = Vxx[:,:,i+1] + (regType == 2 ? lambda*eye(n) : 0)

        Qux_reg = cxu[:,:,i]'   + fu[:,:,i]'Vxx_reg*fx[:,:,i]
        if !isempty(fxu)
            Qux_reg = Qux_reg + fxuVx
        end

        QuuF = cuu[:,:,i]  + fu[:,:,i]'Vxx_reg*fu[:,:,i] + (regType == 1 ? lambda*eye(m) : 0)

        if !isempty(fuu)
            QuuF = QuuF + fuuVx
        end

        @end_backward_pass
    end

    return diverge, Vx, Vxx, k, K, dV
end


function back_pass{T}(cx,cu,cxx::AbstractArray{T,2},cxu,cuu,fx::AbstractArray{T,3},fu,fxx,fxu,fuu,lambda,regType,lims,u) # quadratic timeinvariant cost, dynamics nonlinear time variant

    @setupQTIC

    for i = N-1:-1:1
        Qu  = cu[:,i]   + fu[:,:,i]'Vx[:,i+1]
        Qx  = cx[:,i]   + fx[:,:,i]'Vx[:,i+1]
        Qux = cxu' + fu[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        if !isempty(fxu)
            fxuVx = vectens(Vx[:,i+1],fxu[:,:,:,i])
            Qux   = Qux + fxuVx
        end

        Quu = cuu + fu[:,:,i]'Vxx[:,:,i+1]*fu[:,:,i]
        if !isempty(fuu)
            fuuVx = vectens(Vx[:,i+1],fuu[:,:,:,i])
            Quu   = Quu + fuuVx
        end

        Qxx = cxx  + fx[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        if !isempty(fxx)
            Qxx = Qxx + vectens(Vx[:,i+1],fxx[:,:,:,i])
        end

        Vxx_reg = Vxx[:,:,i+1] + (regType == 2 ? lambda*eye(n) : 0)

        Qux_reg = cxu'   + fu[:,:,i]'Vxx_reg*fx[:,:,i]
        if !isempty(fxu)
            Qux_reg = Qux_reg + fxuVx
        end

        QuuF = cuu  + fu[:,:,i]'Vxx_reg*fu[:,:,i] + (regType == 1 ? lambda*eye(m) : 0)

        if !isempty(fuu)
            QuuF = QuuF + fuuVx
        end

        @end_backward_pass
    end

    return diverge, Vx, Vxx, k, K, dV
end

function back_pass{T}(cx,cu,cxx::AbstractArray{T,2},cxu,cuu,fx::AbstractArray{T,3},fu,lambda,regType,lims,u) # quadratic timeinvariant cost, linear time variant dynamics
    @setupQTIC

    for i = N-1:-1:1
        Qu  = cu[:,i]   + fu[:,:,i]'Vx[:,i+1]
        Qx  = cx[:,i]   + fx[:,:,i]'Vx[:,i+1]
        Qux = cxu' + fu[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        Quu = cuu + fu[:,:,i]'Vxx[:,:,i+1]*fu[:,:,i]
        Qxx = cxx  + fx[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        Vxx_reg = Vxx[:,:,i+1] + (regType == 2 ? lambda*eye(n) : 0)
        Qux_reg = cxu'   + fu[:,:,i]'Vxx_reg*fx[:,:,i]
        QuuF = cuu  + fu[:,:,i]'Vxx_reg*fu[:,:,i] + (regType == 1 ? lambda*eye(m) : 0)

        @end_backward_pass
    end

    return diverge, Vx, Vxx, k, K, dV
end

function back_pass{T}(cx,cu,cxx::AbstractArray{T,2},cxu,cuu,fx::AbstractMatrix{T},fu,lambda,regType,lims,u) # cost quadratic and cost and LTI dynamics

    m  = size(u,1)
    n = size(fx,1)
    N  = length(cx) ÷ n

    cx    = reshape(cx,  (n, N))
    cu    = reshape(cu,  (m, N-1))
    cxx   = reshape(cxx, (n, n))
    cxu   = reshape(cxu, (n, m))
    cuu   = reshape(cuu, (m, m))

    k     = zeros(m,N-1)
    K     = zeros(m,n,N-1)
    Vx    = zeros(n,N)
    Vxx   = zeros(n,n,N)
    dV    = [0., 0.]

    Vx[:,N]     = cx[:,N]
    Vxx[:,:,N]  = cxx

    diverge  = 0
    for i = N-1:-1:1
        Qu  = cu[:,i]      + fu'Vx[:,i+1]
        Qx  = cx[:,i]      + fx'Vx[:,i+1]
        Qux = cxu'  + fu'Vxx[:,:,i+1]*fx

        Quu = cuu   + fu'Vxx[:,:,i+1]*fu
        Qxx = cxx   + fx'Vxx[:,:,i+1]*fx
        Vxx_reg = Vxx[:,:,i+1] + (regType == 2 ? lambda*eye(n) : 0)
        Qux_reg = cxu'   + fu'Vxx_reg*fx

        QuuF = cuu  + fu'Vxx_reg*fu + (regType == 1 ? lambda*eye(m) : 0)

        @end_backward_pass
    end

    return diverge, Vx, Vxx, k, K, dV
end


function graphics(x...)
    return 0
end
