"""
    `x, u, traj_new, Vx, Vxx, cost, trace = iLQGkl(f,costfun,df, x0, u0, traj_prev;
        constrain_per_step = false,
        kl_step            = 0,
        lims               = [],                    # Control signal limits ::Matrix ∈ R(m,2)
        tol_fun            = 1e-7,
        tol_grad           = 1e-4,
        max_iter           = 50,
        print_head         = 10,                    # Print headers this often
        print_period       = 1,                     # Print this often
        reduce_ratio_min   = 0,                     # Not used ATM
        diff_fun           = -,
        verbosity          = 2,                     # ∈ (0,3)
        plot_fun           = x->0,                  # Not used
        cost               = [],                    # Supply if pre-rolled trajectory supplied
        ηbracket           = [1e-8,1,1e16],         # dual variable bracket [min_η, η, max_η]
        del0               = 0.0001,                # Start of dual variable increase
        gd_alpha           = 0.01                   # Step size in GD (ADAMOptimizer) when constrain_per_step is true
        )`

Solves the iLQG problem with constraints on control signals `lims` and bound on the KL-divergence `kl_step` from the old trajectory distribution `traj_prev::GaussianPolicy`. 
"""
function iLQGkl(f,costfun,df, x0, u0, traj_prev;
    constrain_per_step = false,
    kl_step            = 0,
    lims               = [],
    tol_fun            = 1e-7,
    tol_grad           = 1e-4,
    max_iter           = 50,
    print_head         = 10,
    print_period       = 1,
    reduce_ratio_min   = 0,
    diff_fun           = -,
    verbosity          = 2,
    plot_fun           = x->0,
    cost               = [],
    ηbracket           = [1e-8,1,1e16], # min_η, η, max_η
    del0               = 0.0001,
    gd_alpha           = 0.01
    )
    debug("Entering iLQG")

    # --- initial sizes and controls
    u            = copy(traj.k) # initial control sequence
    n            = size(x0, 1) # dimension of state vector
    m,N          = size(u) # dimension of control vector and number of state transitions
    traj_new     = GaussianPolicy(Float64)
    traj_prev.k *= 0 # We are adding new k to u, so must set this to zero for correct kl calculations
    ηbracket     = copy(ηbracket) # Because we do changes in this Array
    if constrain_per_step
        ηbracket = ηbracket.*ones(1,N)
        kl_step = kl_step*ones(N)
    end
    η = view(ηbracket,2,:)

    # --- initialize trace data structure
    trace = [Trace() for i in 1:min( max_iter+1,1e6)]
    trace[1].iter,trace[1].λ,trace[1].dλ = 1,λ,dλ

    # --- initial trajectory
    debug("Setting up initial trajectory")
    if size(x0,2) == 1 # only initial state provided
        diverge = true
        x,u,cost,_ = forward_pass(traj_new,x0[:,1],u,[],1,f,fT, lims,diff_fun)
        debug("# simplistic divergence test")
        if !all(abs.(x) .< 1e8)
            @printf("\nEXIT: Initial control sequence caused divergence\n")
            return
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

    # constants, timers, counters
    Δcost              = 0.
    expected_reduction = 0.
    divergence         = 0.
    step_mult          = 1.
    iter               = 0
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
    xnew,unew,costnew,sigmanew = Matrix{Float64}(0,0),Matrix{Float64}(0,0),Vector{Float64}(0),Matrix{Float64}(0,0)

    dV               = Vector{Float64}()
    reduce_ratio     = 0.
    kl_cost_terms    = (dkl(traj_prev), ηbracket) # This tuple is sent into back_pass, elements in ηbracket are mutated.
    for iter = 1:(constrain_per_step ? 0 : max_iter)
        trace[iter].iter = iter

        # ====== STEP 2: backward pass, compute optimal control law and cost-to-go
        back_pass_done = false
        while !back_pass_done
            tic()
            # debug("Entering back_pass with η=$ηbracket")
            diverge, traj_new,Vx, Vxx,dV = if linearsys
                back_pass(cx,cu,cxx,cxu,cuu,fx,fu,0, regType, lims,x,u,true,kl_cost_terms) # Set λ=0 since we use η
            else
                back_pass(cx,cu,cxx,cxu,cuu,fx,fu,fxx,fxu,fuu,0, regType, lims,x,u,true,kl_cost_terms) # Set λ=0 since we use η
            end
            trace[iter].time_backward = toq()

            if diverge > 0
                ηbracket[2] .+= del0 # TODO: modify η here
                del0 *= 2
                if verbosity > 2; println("Cholesky failed at timestep $diverge. η-bracket: ", ηbracket); end
                if ηbracket[2] >  0.99ηbracket[3] #  terminate ?
                    verbosity > 0 && @printf("\nEXIT: η > ηmax (back_pass failed)\n")
                    break
                end
                continue
            end
            debug("Back pass done")
            back_pass_done = true
        end

        #  check for termination due to small gradient
        g_norm = mean(maximum(abs.(traj_new.k) ./ (abs.(u)+1),1))
        trace[iter].grad_norm = g_norm

        # ====== STEP 3: Forward pass
        if back_pass_done
            tic()
            # debug("#  entering forward_pass")
            xnew,unew,costnew,sigmanew = forward_pass(traj_new, x0[:,1] ,u, x,1,f,costfun, lims, diff_fun)

            Δcost    = sum(cost) - sum(costnew)
            expected_reduction = -(dV[1] + dV[2])

            reduce_ratio = if expected_reduction > 0
                Δcost/expected_reduction
            else
                warn("negative expected reduction: should not occur")
                sign(Δcost)
            end
            ηbracket, satisfied, divergence = calc_η(xnew,x,sigmanew,ηbracket, traj_new, traj_prev, kl_step)
            trace[iter].time_forward = toq()
            debug("Forward pass done: η: $ηbracket")
        end


        # ====== STEP 4: accept step (or not), print status

        #  print headings
        if verbosity > 1 && iter % print_period == 0
            if last_head == print_head
                last_head = 0
                @printf("%-12s", "iteration     cost    reduction     expected    gradient    log10(η)    divergence      entropy\n")
            end
            @printf("%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f%-12.3g%-12.3g\n",
            iter, sum(costnew), Δcost, expected_reduction, g_norm, log10(mean(η)), mean(divergence), entropy(traj_new))
            last_head += 1
        end
        #  update trace
        trace[iter].λ            = λ
        trace[iter].dλ           = dλ
        trace[iter].alpha        = 1
        trace[iter].improvement  = Δcost
        trace[iter].cost         = sum(costnew)
        trace[iter].reduce_ratio = reduce_ratio
        trace[iter].divergence   = mean(divergence)
        trace[iter].η            = ηbracket[2]

        # Termination checks
        if g_norm <  tol_grad && divergence-kl_step > 0 # In this case we're only going to get even smaller gradients and might as well quit
            verbosity > 0 && @printf("\nEXIT: gradient norm < tol_grad while constraint violation too large\n")
            break
        end
        if satisfied # KL-constraint is satisfied and we're happy (at least if Δcost is positive)
            plot_fun(x)
            verbosity > 0 && @printf("\nSUCCESS: abs(KL-divergence) < kl_step\n")
            break
        end
        if ηbracket[2] >  0.99ηbracket[3]
            verbosity > 0 && @printf("\nEXIT: η > ηmax\n")
            break
        end
        graphics(x,u,cost,K,Vx,Vxx,fx,fxx,fu,fuu,trace[1:iter],0)
    end

    if constrain_per_step # This implements the gradient descent procedure for η
        optimizer = ADAMOptimizer(kl_step, α=gd_alpha)
        for iter = 1:max_iter
            diverge = 1
            del = del0*ones(N)
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
                    if verbosity > 2; println("Cholesky failed at timestep $diverge. η-bracket: ", mean(η)); end
                    if all(ηbracket[2,:] .>  0.99ηbracket[3,:])
                        # TODO: This termination criteria could be improved
                        verbosity > 0 && @printf("\nEXIT: η > ηmax\n")
                        break
                    end
                end
            end
            k, K = traj_new.k, traj_new.K

            xnew,unew,costnew,sigmanew = forward_pass(traj_new, x0[:,1] ,u, x,1,f,costfun, lims, diff_fun)
            Δcost                 = sum(cost) - sum(costnew)
            expected_reduction    = -(dV[1] + dV[2])
            reduce_ratio          = Δcost/expected_reduction
            divergence            = kl_div_wiki(xnew,x,sigmanew, traj_new, traj_prev)
            constraint_violation  = divergence - kl_step
            lη                    = log.(η) # Run GD in log-space (much faster)
            η                    .= exp(optimizer(lη, -constraint_violation, iter))
            η                    .= clamp.(η, ηbracket[1,:], ηbracket[3,:])
            g_norm                = mean(maximum(abs.(k) ./ (abs.(u)+1),1))
            trace[iter].grad_norm = g_norm
            if all(abs.(constraint_violation) .< 0.1*kl_step) # TODO: This almost never happens, gradient on past time indices should be influenced by future constraint_violations
                satisfied = true
                break
            end
            if verbosity > 1 && iter % print_period == 0
                if last_head == print_head
                    last_head = 0
                    @printf("%-12s", "iteration     cost    reduction     expected    log10(η)    divergence      entropy\n")
                end
                @printf("%-12d%-12.6g%-12.3g%-12.3g%-12.1f%-12.3g%-12.3g\n",
                iter, sum(costnew), Δcost, expected_reduction, mean(log10.(η)), mean(divergence), entropy(traj_new))
                last_head += 1
            end
        end
    end

    iter ==  max_iter &&  verbosity > 0 && @printf("\nEXIT: Maximum iterations reached.\n")
    if Δcost > 0 # In this case we made an improvement under the model and accept the changes
        x,u,cost  = xnew,unew,costnew
        traj_new.k = copy(u) # TODO: is this a good idea? maybe only accept changes if kl satisfied?
    else
        verbosity > 0 && println("Cost increased, did not accept changes to u")
    end

    any((divergence .> kl_step) .& (abs.(divergence - kl_step) .> 0.1*kl_step)) && warn("KL divergence too high for some time steps when done")
    verbosity > 0 && print_timing(trace,iter,t_start,cost,g_norm,mean(ηbracket[2,:]))

    return x, u, traj_new, Vx, Vxx, cost, trace
end

# TODO: Implement adaptive kl-step
# TODO: Implement controller visualization
