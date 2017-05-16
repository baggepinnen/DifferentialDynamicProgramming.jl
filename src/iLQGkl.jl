function iLQGkl(f,fT,df, x0, u0, traj_prev;
    lims               = [],
    alpha              = logspace(0,-3,11),
    tol_fun            = 1e-7,
    tol_grad           = 1e-4,
    max_iter           = 500,
    λ                  = 1,
    dλ                 = 1,
    λfactor            = 1.6,
    λmax               = 1e10,
    λmin               = 1e-6,
    regType            = 1,
    reduce_ratio_min   = 0,
    diff_fun           = -,
    verbosity          = 2,
    plot_fun           = x->0,
    cost               = [],
    kl_step            = 0,
    ηbracket           = [1e-8,1,1e16], # min_η, η, max_η
    constrain_per_step = false,
    del0               = 0.0001
    )
    debug("Entering iLQG")

    # --- initial sizes and controls
    n   = size(x0, 1)          # dimension of state vector
    m   = size(u0, 1)          # dimension of control vector
    N   = size(u0, 2)          # number of state transitions
    u   = u0                   # initial control sequence
    traj_new  = GaussianPolicy(Float64)
    traj_prev.k *= 0 # We are adding new k to u, so must set this to zero for correct kl calculations
    if constrain_per_step
        ηbracket = ηbracket.*ones(1,N)
        kl_step = kl_step*ones(N)
    end

    # kl_step *= N # constrain per step not yet supported, changed to mean in kl_div
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
    step_mult          = 1.
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
        del = constrain_per_step ? del0*ones(N) : del0
        kl_cost_terms = (dkl(traj_prev), ηbracket) # TODO: Magic tuple
        while !back_pass_done
            tic()
            # @show size(cxkl),size(cukl),size(cxxkl),size(cuukl),size(cxukl)
            # @show size(cx),size(cu),size(cxx),size(cuu),size(cxu)
            # debug("Entering back_pass with η=$ηbracket")
            diverge, traj_new,Vx, Vxx,dV = if linearsys
                back_pass(cx,cu,cxx,cxu,cuu,fx,fu,0, regType, lims,x,u,true,kl_cost_terms) # Set λ=0 since we use η
            else
                back_pass(cx,cu,cxx,cxu,cuu,fx,fu,fxx,fxu,fuu,0, regType, lims,x,u,true,kl_cost_terms) # Set λ=0 since we use η
            end
            trace[iter].time_backward = toq()

            if diverge > 0
                delind = constrain_per_step ? diverge : 1
                ηbracket[2,delind] .+= del[delind] # TODO: modify η here
                del[delind] *= 2
                if verbosity > 2; println("Cholesky failed at timestep $diverge. η-bracket: ", ηbracket); end
                if all(ηbracket[2,:] .>  0.99ηbracket[3,:]) #  terminate ?
                    verbosity > 0 && @printf("\nEXIT: η > ηmax\n")
                    break
                end
                continue
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

        # ====== STEP 3: Forward pass
        if back_pass_done
            tic()
            xnew,unew,costnew,sigmanew = Matrix{Float64}(0,0),Matrix{Float64}(0,0),Vector{Float64}(0),Matrix{Float64}(0,0)
            # debug("#  entering forward_pass")
            alphai = 1
            xnew,unew,costnew,sigmanew = forward_pass(traj_new, x0[:,1] ,u, x,alphai,f,fT, lims, diff_fun)

            Δcost    = sum(cost) - sum(costnew)
            expected_reduction = -alphai*(dV[1] + alphai*dV[2])

            reduce_ratio = if expected_reduction > 0
                Δcost/expected_reduction
            else
                warn("negative expected reduction: should not occur")
                sign(Δcost)
            end
            ηbracket, satisfied, divergence = calc_η(xnew,x,sigmanew,ηbracket, traj_new, traj_prev, kl_step)
            trace[iter].time_forward = toq()
        end

        if constrain_per_step # This implements the gradient descent procedure for η
            optimizer = ADAMOptimizer(kl_step, α=0.01)
            for gd_iter = 1:100
                diverge = 1
                del = constrain_per_step ? del0*ones(N) : del0
                while diverge > 0
                    diverge, traj_new,Vx, Vxx,dV = if linearsys
                        back_pass(cx,cu,cxx,cxu,cuu,fx,fu,0, regType, lims,x,u,true,kl_cost_terms)
                    else
                        back_pass(cx,cu,cxx,cxu,cuu,fx,fu,fxx,fxu,fuu,0, regType, lims,x,u,true,kl_cost_terms)
                    end
                    if diverge > 0
                        delind = diverge
                        ηbracket[2,delind] .+= del[delind] # TODO: modify η here
                        del[delind] *= 2
                        if verbosity > 2; println("Cholesky failed at timestep $diverge. η-bracket: ", ηbracket); end
                        if all(ηbracket[2,:] .>  0.99ηbracket[3,:]) #  terminate ?
                            verbosity > 0 && @printf("\nEXIT: η > ηmax\n")
                            break
                        end
                    end
                end
                k, K = traj_new.k, traj_new.K

                xnew,unew,costnew,sigmanew = forward_pass(traj_new, x0[:,1] ,u, x,alphai,f,fT, lims, diff_fun)
                divergence    = kl_div_wiki(xnew,x,sigmanew, traj_new, traj_prev)
                constraint_violation = divergence - kl_step
                if all(abs.(constraint_violation) .< 0.1*kl_step)
                    break
                end
                optimizer(η, -constraint_violation, iter)
                η .= clamp.(η, ηbracket[1,:], ηbracket[3,:])
                println(η')
            end
        end
        # ====== STEP 4: accept step (or not), print status

        #  print headings
        if verbosity > 1
            if last_head == print_head
                last_head = 0
                @printf("%-12s", "iteration     cost    reduction     expected    gradient    log10(λ)    log10(η)    divergence      entropy\n")
            end
            @printf("%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f%-12.1f%-12.3g%-12.3g\n",
            iter, sum(cost), Δcost, expected_reduction, g_norm, log10(λ), log10(mean(η)), mean(divergence), entropy(traj_new))
            last_head += 1
        end

        if satisfied
            plot_fun(x)
            verbosity > 0 &&  @printf("\nSUCCESS: abs(KL-divergence) < kl_step\n")
            break

        else #  no cost improvement
            alphai =  NaN
            push!(trace, Trace())
        end
        if all(ηbracket[2,:] .>  0.99ηbracket[3,:]) #  terminate ?
            verbosity > 0 && @printf("\nEXIT: η > ηmax\n")
            break
        end
        #  update trace
        trace[iter].λ            = λ
        trace[iter].dλ           = dλ
        trace[iter].alpha        = alphai
        trace[iter].improvement  = Δcost
        trace[iter].cost         = sum(costnew)
        trace[iter].reduce_ratio = reduce_ratio
        trace[iter].divergence   = mean(divergence)
        trace[iter].η            = mean(η)
        graphics(x,u,cost,K,Vx,Vxx,fx,fxx,fu,fuu,trace[1:iter],0)
    end

    iter ==  max_iter &&  verbosity > 0 && @printf("\nEXIT: Maximum iterations reached.\n")
    x,u,cost  = xnew,unew,costnew
    traj_new.k = copy(u) # TODO: is this a good idea? maybe only accept changes if kl satisfied?

    any((divergence .> kl_step) .& (abs.(divergence - kl_step) .> 0.1*kl_step)) && warn("KL divergence too high when done")
    verbosity > 0 && print_timing(trace,iter,t_start,cost,g_norm,mean(ηbracket[2,:]))

    return x, u, traj_new, Vx, Vxx, cost, trace
end
