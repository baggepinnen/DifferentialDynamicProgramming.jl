"""
Calculate the Q terms related to the KL-constraint.
Qtt is [Qxx Qxu; Qux Quu]
Qt is [Qx; Qu]
These terms should be added to the Q terms calculated in the backwards pass to produce the final Q terms.
This Function should be called from within the backwards_pass Function or just prior to it to adjust the cost derivative matrices.
"""
function dkl(traj_new)
    isempty(traj_new) && (return (0,0,0,0,0))
    debug("Calculating KL cost addition terms")
    m,n,T  = traj_new.m,traj_new.n,traj_new.T
    cx,cu,cxx,cuu,cxu = zeros(n,T),zeros(m,T),zeros(n,n,T),zeros(m,m,T),zeros(n,m,T)
    for t in 1:T
        K, k       = traj_new.K[:,:,t], traj_new.k[:,t]
        Σi         = traj_new.Σi[:,:,t]
        cx[:,t]    = K'*Σi*k
        cu[:,t]    = -Σi*k
        cxx[:,:,t] = K'*Σi*K
        cuu[:,:,t] = Σi
        cxu[:,:,t] = -K'Σi#TODO: maybe -Σi*K? Does fuck up array dims later, https://github.com/cbfinn/gps/blob/master/python/gps/algorithm/traj_opt/traj_opt_lqr_python.py#L355
    end
    return cx,cu,cxx,cxu,cuu
end


function KLmv(Σi,K,k)
    M =
    [K'*Σi*K  -K'*Σi;
    -Σi*K    Σi ]
    K', Σi, k
    v = [K'*Σi*k  -Σi*k]
    M,v
end

function kl_div(xnew,unew, Σ_new, new_traj::GaussianPolicy, prev_traj::GaussianPolicy)
    (isempty(new_traj) || isempty(prev_traj)) && (return 0)
    μ_new = [xnew; unew]
    T     = new_traj.T
    # m     = size(new_traj.fu,1)
    kldiv = zeros(T)
    for t = 1:T
        μt    = μ_new[:,t]
        Σt    = Σ_new[:,:,t]
        Kp    = prev_traj.K[:,:,t]
        Kn    = new_traj.K[:,:,t]
        kp    = prev_traj.k[:,t]
        kn    = new_traj.k[:,t]
        Σp    = prev_traj.Σ[:,:,t]
        Σn    = new_traj.Σ[:,:,t]
        Σip   = prev_traj.Σi[:,:,t]
        Σin   = new_traj.Σi[:,:,t]
        Mp,vp = KLmv(Σip,Kp,kp)
        Mn,vn = KLmv(Σin,Kn,kn)
        cp    = prev_traj.dV[2]
        cn    = new_traj.dV[2]

        kldiv[t] = -0.5μt'(Mn-Mp)*μt -  μt'(vn-vp) - cn + cp -0.5sum(Σt*(Mn-Mp)) -0.5logdet(Σn) + 0.5logdet(Σp)
        kldiv[t] = max(0,kldiv[t])
    end
    return kldiv
end


function kl_div_wiki(xnew,xold, Σ_new, new_traj::GaussianPolicy, prev_traj::GaussianPolicy)
    μ_new = xnew-xold# [xnew; unew] verkar inte som att unew behövs??
    T,m     = new_traj.T, new_traj.m
    kldiv = zeros(T)
    for t = 1:T
        μt     = μ_new[:,t] # TODO: why is traj mean not compared to old traj mean?
        Σt     = Σ_new[:,:,t]
        Kp     = prev_traj.K[:,:,t]
        Kn     = new_traj.K[:,:,t]
        kp     = prev_traj.k[:,t]
        kn     = new_traj.k[:,t]
        Σp     = prev_traj.Σ[:,:,t]
        Σn     = new_traj.Σ[:,:,t]
        Σip    = prev_traj.Σi[:,:,t]
        Σin    = new_traj.Σi[:,:,t]
        dim    = m
        k_diff = kp-kn
        K_diff = Kp-Kn
        try
            kldiv[t] = 1/2 * (trace(Σip*Σn) + k_diff⋅(Σip*k_diff) - dim + logdet(Σp) - logdet(Σn) )
            kldiv[t] +=  1/2 *( μt'K_diff'Σip*K_diff*μt + trace(K_diff'Σip*K_diff*Σt) )[1]
            kldiv[t] += k_diff ⋅ (Σip*K_diff*μt)
        catch e
            println(e)
            @show Σip, Σin, Σp, Σn
            return Inf
        end
    end
    kldiv = max.(0,kldiv)
    return kldiv
end


entropy(traj::GaussianPolicy) = mean(logdet(traj.Σ[:,:,t])/2 for t = 1:traj.T) + traj.m*log(2π*e)/2
# TODO: Calculate Σ in the forwards pass, requires covariance of forward dynamics model. Is this is given by the Pkn matrix from the Kalman model?


"""
new_η, satisfied, divergence = calc_η(xnew,xold,sigmanew,η, traj_new, traj_prev, kl_step)
This Function caluculates the step size
"""
function calc_η(xnew,xold,sigmanew,ηbracket, traj_new, traj_prev, kl_step::Number)
    kl_step > 0 || (return (1., true,0))

    divergence    = kl_div_wiki(xnew,xold,sigmanew, traj_new, traj_prev) |> mean
    constraint_violation = divergence - kl_step
    # Convergence check - constraint satisfaction.
    satisfied = abs(constraint_violation) < 0.1*kl_step # allow some small constraint violation
    if satisfied
        debug(@sprintf("KL: %12.7f / %12.7f, converged",  divergence, kl_step))
    else
        if constraint_violation < 0 # η was too big.
            ηbracket[3] = ηbracket[2]
            ηbracket[2] = max(geom(ηbracket), 0.1*ηbracket[3])
            debug(@sprintf("KL: %12.4f / %12.4f, η too big, new η: (%-5.3g < %-5.3g < %-5.3g)",  divergence, kl_step, ηbracket...))
        else # η was too small.
            ηbracket[1] = ηbracket[2]
            ηbracket[2] = min(geom(ηbracket), 10.0*ηbracket[1])
            debug(@sprintf("KL: %12.4f / %12.4f, η too small, new η: (%-5.3g < %-5.3g < %-5.3g)",  divergence, kl_step, ηbracket...))
        end
    end
    return ηbracket, satisfied, divergence
end

function calc_η(xnew,xold,sigmanew,ηbracket, traj_new, traj_prev, kl_step::AbstractVector)
    any(kl_step .> 0) || (return (1., true,0))

    divergence    = kl_div_wiki(xnew,xold,sigmanew, traj_new, traj_prev)
    if !isa(kl_step,AbstractVector)
        divergence = mean(divergence)
    end
    constraint_violation = divergence - kl_step
    # Convergence check - constraint satisfaction.
    satisfied = all(abs.(constraint_violation) .< 0.1*kl_step) # allow some small constraint violation
    if satisfied
        debug(@sprintf("KL: %12.7f / %12.7f, converged",  mean(divergence), mean(kl_step)))
    else
        too_big = constraint_violation .< 0

        ηbracket[3,too_big] = ηbracket[2,too_big]
        ηbracket[2,too_big] = max.(geom(ηbracket[:,too_big]), 0.1*ηbracket[3,too_big])

        ηbracket[1,!too_big] = ηbracket[2,!too_big]
        ηbracket[2,!too_big] = min.(geom(ηbracket[:,!too_big]), 10.0*ηbracket[1,!too_big])
    end
    return ηbracket, satisfied, divergence
end
geom(ηbracket::AbstractMatrix) = sqrt.(ηbracket[1,:].*ηbracket[3,:])
geom(ηbracket::AbstractVector) = sqrt(ηbracket[1]*ηbracket[3])

# using Base.Test
# n,m,T = 1,1,1
#
# traj_new  = GaussianDist(Float64,T,n,m)
# traj_old  = GaussianDist(Float64,T,n,m)
# xnew = zeros(n,T)
# unew = zeros(m,T)
# Σnew = cat(3,[eye(n+m) for t=1:T]...)
# @test kl_div_wiki(xnew,unew, Σnew, traj_new, traj_old) == 0
#
# traj_new.μu = ones(m,T)
# kl_div_wiki(xnew,unew, Σnew, traj_new, traj_old)
#
# traj_new.μx = ones(m,T)
# kl_div_wiki(xnew,unew, Σnew, traj_new, traj_old)
#
# traj_new.Σ .*=2
# kl_div_wiki(xnew,unew, Σnew, traj_new, traj_old)
