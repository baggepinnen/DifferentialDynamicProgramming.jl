function iLQGkl(f,fT,df, x0, u0, traj_prev;
    lims             = [],
    alpha            = logspace(0,-3,11),
    tol_fun          = 1e-7,
    tol_grad         = 1e-4,
    max_iter         = 500,
    λ                = 1,
    dλ               = 1,
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
    kl_step          = 0,
    ηbracket         = [1e-8,1,1e16] # min_η, η, max_η
    )
    debug("Entering iLQG")

    # --- initial sizes and controls
    n   = size(x0, 1)          # dimension of state vector
    m   = size(u0, 1)          # dimension of control vector
    N   = size(u0, 2)          # number of state transitions
    u   = u0                   # initial control sequence
    traj_new  = GaussianPolicy(Float64)

    kl_step *= N # constrain per step not yet supported
    # --- initialize trace data structure
    trace = [Trace() for i in 1:min( max_iter+1,1e6)]
    trace[1].iter,trace[1].λ,trace[1].dλ = 1,λ,dλ

    # --- initial trajectory
    debug("Setting up initial trajectory")
    if size(x0,2) == 1 # only initial state provided
        diverge = true
        for alphai ∈ alpha
            debug("# test different backtracing parameters alpha and break loop when first succeeds")
            x,un,cost,_ = forward_pass(traj_new,x0[:,1],alphai*u,[],1,f,fT, lims,diff_fun)
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

    trace[1].cost = sum(cost)
    #     plot_fun(x) # user plotting

    if diverge
        if verbosity > 0
            @printf("\nEXIT: Initial control sequence caused divergence\n")
        end
        return
    end

    # constants, timers, counters
    Δcost              = 0.
    expected_reduction = 0.
    divergence         = 0.
    print_head         = 10 # print headings every print_head lines
    last_head          = print_head
    g_norm             = Vector{Float64}()
    Vx = Vxx           = emptyMat3(Float64)
    xnew,unew,costnew  = similar(x),similar(u),Vector{Float64}(N)
    t_start            = time()
    verbosity > 0 && @printf("\n---------- begin iLQG ----------\n")
    satisfied          = false

    # ====== STEP 1: differentiate dynamics and cost along new trajectory
    trace[1].time_derivs = @elapsed fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu = df(x, u)

    # Determine what kind of system we are dealing with
    linearsys = isempty(fxx) && isempty(fxu) && isempty(fuu); debug("linear system: $linearsys")

    for iter = 1:max_iter
        trace[iter].iter = iter
        dV               = Vector{Float64}()
        reduce_ratio     = 0.

        # ====== STEP 2: backward pass, compute optimal control law and cost-to-go

        back_pass_done = false
        while !back_pass_done
            tic()

            cxkl,cukl,cxxkl,cxukl,cuukl = dkl(traj_new)
            if ndims(cxx) == 2 && size(cxxkl) != () # TODO: If the special case ndims(cxx) == 2 it will be promoted to 3 dims and another back_pass method will be called, move the addition of costs and if statement into dkl
                cxxkl,cuukl,cxukl = cxxkl[:,:,1],cuukl[:,:,1],cxukl[:,:,1]
            end
            # @show size(cxkl),size(cukl),size(cxxkl),size(cuukl),size(cxukl)
            # @show size(cx),size(cu),size(cxx),size(cuu),size(cxu)
            η = ηbracket[2]
            cxi,cui,cxxi,cxui,cuui = cx./η.+cxkl, cu./η.+cukl, cxx./η.+cxxkl, cxu./η.+cxukl, cuu./η.+cuukl
            debug("Entering back_pass with η=$ηbracket")
            diverge, traj_new,Vx, Vxx,dV = if linearsys
                back_pass(cxi,cui,cxxi,cxui,cuui,fx,fu,0, regType, lims,x,u,true) # Set λ=0 since we use η
            else
                back_pass(cxi,cui,cxxi,cxui,cuui,fx,fu,fxx,fxu,fuu,0, regType, lims,x,u,true) # Set λ=0 since we use η
            end
            warn("It seems modifying η has very little effect on the KL-div. Investigate this")
            trace[iter].time_backward = toq()

            if diverge > 0
                ηbracket[2] *= 2 # TODO: modify η here
                verbosity > 2 && @printf("Cholesky failed at timestep %d. η-bracket: %-12.3g\n",diverge, ηbracket)
                if ηbracket[2] >  0.99ηbracket[3] #  terminate ?
                    verbosity > 0 && @printf("\nEXIT: η > ηmax\n")
                    break
                end
            end
            back_pass_done = true
        end
        k, K = traj_new.k, traj_new.K
        #  check for termination due to small gradient
        g_norm = mean(maximum(abs.(k) ./ (abs.(u)+1),1))
        trace[iter].grad_norm = g_norm
        if g_norm <  tol_grad && satisfied
            verbosity > 0 && @printf("\nSUCCESS: gradient norm < tol_grad\n")
            break
        end

        # ====== STEP 3: line-search to find new control sequence, trajectory, cost
        if back_pass_done
            tic()
            xnew,unew,costnew,sigmanew = Matrix{Float64}(0,0),Matrix{Float64}(0,0),Vector{Float64}(0),Matrix{Float64}(0,0)
            debug("#  entering forward_pass")
            # for alphai = alpha # TODO: Maybe this linesearch is replaced by the search for η?
            alphai = 1
            xnew,unew,costnew,sigmanew = forward_pass(traj_new, x0[:,1] ,u, x,alphai,f,fT, lims, diff_fun)
            Δcost    = sum(cost) - sum(costnew)
            expected_reduction = -alphai*(dV[1] + alphai*dV[2])
            # expected_reduction /= ηbracket[2] # TODO: This should not be needed since dV is affected by Quu
            reduce_ratio = if expected_reduction > 0
                Δcost/expected_reduction
            else
                warn("negative expected reduction: should not occur")
                sign(Δcost)
            end
            ηbracket, satisfied, divergence = calc_η(xnew,unew,sigmanew,ηbracket, traj_new, traj_prev, kl_step)
            if satisfied; break;end
            # end
            trace[iter].time_forward = toq()
        end

        # ====== STEP 4: accept step (or not), print status

        #  print headings
        if verbosity > 1
            if last_head == print_head
                last_head = 0
                @printf("%-12s", "iteration     cost    reduction     expected    gradient    log10(λ)    η    divergence\n")
            end
            @printf("%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f%-12.3g%-12.3g\n",
            iter, sum(cost), Δcost, expected_reduction, g_norm, log10(λ), η, divergence)
            last_head += 1
        end

        if satisfied # TODO: I added satisfied here, verify if this is reasonable
            #  accept changes
            plot_fun(x)

            verbosity > 0 &&  @printf("\nSUCCESS: abs(KL-divergence) < kl_step\n")
            break

        else #  no cost improvement
            alphai =  NaN
            push!(trace, Trace())
        end
        #  update trace
        trace[iter].λ            = λ
        trace[iter].dλ           = dλ
        trace[iter].alpha        = alphai
        trace[iter].improvement  = Δcost
        trace[iter].cost         = sum(cost)
        trace[iter].reduce_ratio = reduce_ratio
        trace[iter].divergence   = divergence
        trace[iter].η            = η
        graphics( plot,x,u,cost,K,Vx,Vxx,fx,fxx,fu,fuu,trace[1:iter],0)
    end

    iter ==  max_iter &&  verbosity > 0 && @printf("\nEXIT: Maximum iterations reached.\n")
    x,u,cost  = xnew,unew,costnew
    traj_new.k = copy(u) # TODO: maybe only accept changes if kl satisfied?

    divergence > kl_step && abs(divergence - kl_step) > 0.1*kl_step && warn("KL divergence too high when done")
    verbosity > 0 && print_timing(trace,iter,t_start,cost,g_norm,λ)

    return x, u, traj_new, Vx, Vxx, cost, trace
end
