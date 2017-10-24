"""
Calculate the Q terms related to the KL-constraint. (Actually, only related to log(p̂(τ)) since the constraint is rewritten as Entropy term and other term dissapears into expectation under p(τ).)
Qtt is [Qxx Qxu; Qux Quu]
Qt is [Qx; Qu]
These terms should be added to the Q terms calculated in the backwards pass to produce the final Q terms.
This Function should be called from within the backwards_pass Function or just prior to it to adjust the cost derivative matrices.
"""
function ∇kl(traj_prev)
    isempty(traj_prev) && (return (0,0,0,0,0))
    debug("Calculating KL cost addition terms")
    m,n,T  = traj_prev.m,traj_prev.n,traj_prev.T
    cx,cu,cxx,cuu,cxu = zeros(n,T),zeros(m,T),zeros(n,n,T),zeros(m,m,T),zeros(m,n,T)
    for t in 1:T
        K, k       = traj_prev.K[:,:,t], traj_prev.k[:,t]
        Σi         = traj_prev.Σi[:,:,t]
        cx[:,t]    = K'*Σi*k
        cu[:,t]    = -Σi*k
        cxx[:,:,t] = K'*Σi*K
        cuu[:,:,t] = Σi
        cxu[:,:,t] = -Σi*K # https://github.com/cbfinn/gps/blob/master/python/gps/algorithm/traj_opt/traj_opt_lqr_python.py#L355
    end
    return cx,cu,cxx,cxu,cuu
end

"""
    This is the inverse of Σₓᵤ
"""
function KLmv(Σi,K,k)
    M =
    [K'*Σi*K  -K'*Σi;
    -Σi*K    Σi ]
    v = [K'*Σi*k;  -Σi*k]
    M,v
end

"""
    This function produces lots of negative values which are clipped by the max(0,kl)
"""
function kl_div(xnew,xold, Σ_new, traj_new, traj_prev)
    (isempty(traj_new) || isempty(traj_prev)) && (return 0)
    μ_new = [xnew-xold; unew]
    T     = traj_new.T
    # m     = traj_new.m
    kldiv = zeros(T)
    for t = 1:T
        μt    = μ_new[:,t]
        Σt    = Σ_new[:,:,t]
        Kp    = traj_prev.K[:,:,t]
        Kn    = traj_new.K[:,:,t]
        kp    = traj_prev.k[:,t]
        kn    = traj_new.k[:,t] + kp # unew must be added here
        Σp    = traj_prev.Σ[:,:,t]
        Σn    = traj_new.Σ[:,:,t]
        Σip   = traj_prev.Σi[:,:,t]
        Σin   = traj_new.Σi[:,:,t]
        Mp,vp = KLmv(Σip,Kp,kp)
        Mn,vn = KLmv(Σin,Kn,kn)
        cp    = .5*kp'Σip*kp
        cn    = .5*kn'Σin*kn

        kldiv[t] = -0.5μt'*(Mn-Mp)*μt -  μt'*(vn-vp) - cn + cp -0.5sum(Σt*(Mn-Mp)) -0.5logdet(Σn) + 0.5logdet(Σp)
        kldiv[t] = max.(0,kldiv[t])
    end
    return kldiv
end

"""
This version seems to be symmetric and positive
"""
function kl_div_wiki(xnew,xold, Σ_new, traj_new, traj_prev)
    μ_new = xnew-xold
    T,m,n     = traj_new.T, traj_new.m, traj_new.n
    kldiv = zeros(T)
    for t = 1:T
        μt     = μ_new[:,t]
        Σt     = Σ_new[1:n,1:n,t]
        Kp     = traj_prev.K[:,:,t]
        Kn     = traj_new.K[:,:,t]
        kp     = traj_prev.k[:,t]
        kn     = traj_new.k[:,t] #traj_new.k[:,t] contains kp already
        Σp     = traj_prev.Σ[:,:,t]
        Σn     = traj_new.Σ[:,:,t]
        Σip    = traj_prev.Σi[:,:,t]
        Σin    = traj_new.Σi[:,:,t]
        dim    = m
        k_diff = kp-kn
        K_diff = Kp-Kn
        try
            kldiv[t] = 1/2 * (trace(Σip*Σn) + k_diff'Σip*k_diff - dim + logdet(Σp) - logdet(Σn) ) # Wikipedia term
            kldiv[t] +=  1/2 *( μt'K_diff'Σip*K_diff*μt + trace(K_diff'Σip*K_diff*Σt) )[1]
            kldiv[t] += k_diff'Σip*K_diff*μt
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
        debug("calc_η: Sum(too big η) = $sum(too_big)")

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
n,m,T = 1,1,3
Σnew = cat(3,[eye(n+m) for t=1:T]...)
Σ = cat(3,[eye(m) for t=1:T]...)
K = zeros(m,n,T)
k = zeros(m,T)

traj_new  = DifferentialDynamicProgramming.GaussianPolicy(T,n,m,K,k,Σ,Σ)
traj_prev  = DifferentialDynamicProgramming.GaussianPolicy(T,n,m,copy(K),copy(k),copy(Σ),copy(Σ))
xnew = zeros(n,T)
xold = zeros(n,T)
unew = zeros(m,T)

kl_div_wiki(xnew,xold, Σnew, traj_new, traj_prev)

traj_new.k = ones(m,T)
traj_prev.k = ones(m,T)
kl_div_wiki(xnew,xold, Σnew, traj_new, traj_prev)
traj_new.k .*= 0

traj_new.K = ones(m,n,T)
kl_div_wiki(xnew,xold, Σnew, traj_new, traj_prev)
traj_new.K .*= 0

traj_new.Σ .*=2
kl_div_wiki(xnew,xold, Σnew, traj_new, traj_prev)


mutable struct ADAMOptimizer{T,N}
    α::T
    β1::T
    β2::T
    ɛ::T
    m::Array{T,N}
    v::Array{T,N}
end

ADAMOptimizer{T,N}(g::AbstractArray{T,N}; α = 0.005,  β1 = 0.9, β2 = 0.999, ɛ = 1e-8, m=zeros(g), v=zeros(g)) = ADAMOptimizer{T,N}(α,  β1, β2, ɛ, m, v)

"""
    (a::ADAMOptimizer{T,N})(Θ::Array{T,N}, g::Array{T,N}, t::Number)

Applies the gradient `g` to the parameters `Θ` (mutating) at iteration `t`
ADAM GD update http://sebastianruder.com/optimizing-gradient-descent/index.html#adam
"""
function (a::ADAMOptimizer{T,N}){T,N}(Θ::AbstractArray{T,N}, g::AbstractArray{T,N}, t)
    a.m .= a.β1 .* a.m .+ (1-a.β1) .* g
    m̂    = a.m ./ (1 - a.β1 ^ t)
    a.v .= a.β2 .* a.v .+ (1-a.β2) .* g.^2
    v̂    = a.v ./ (1 - a.β2 ^ t)
    # @show size(Θ), size(m̂), size(v̂)
    Θ  .-= a.α .* m̂ ./ (sqrt.(v̂) .+ a.ɛ)
end
