
"""
Compute KL divergence between new and previous trajectory
distributions.

μ_new: (n+m)×T, mean of new trajectory distribution (xnew, unew).
Σ_new: n×n×T , variance of new trajectory distribution.
"""
function kl_div(xnew,unew, Σ_new, new_traj::GaussianDist, prev_traj::GaussianDist)
    μ_new = [xnew; unew]
    T     = new_traj.T
    m     = size(new_traj.K,1)
    kldiv = zeros(T)
    for t = 1:T
        μt    = μ_new[:,t]
        Σt    = Σ_new[:,:,t]
        Kp    = prev_traj.fx[:,:,t]
        Kn    = new_traj.fx[:,:,t]
        kp    = prev_traj.μu[:,t]
        kn    = new_traj.μu[:,t]
        Σp    = prev_traj.Σ[:,:,t]
        Σn    = new_traj.Σ[:,:,t]
        Σip   = inv(Σip)
        Σin   = inv(Σin)
        Mp,vp = KLmv(Σip,Kp,kp)
        Mn,vn = KLmv(Σin,Kn,kn)
        cp    = prev_traj.dV[2]
        cn    = new_traj.dV[2]

        kldiv[t] = -0.5μt'(Mn-Mp)*μt -  μt'(vn-vp) - cn + cp -0.5sum(Σt*(Mn-Mp)) -0.5logdet(Σn) + 0.5logdet(Σp)
        kldiv[t] = max(0,kldiv[t])
    end
    return sum(kldiv)
end


function kl_div_wiki(xnew,unew, Σ_new, new_traj::GaussianDist, prev_traj::GaussianDist)
    μ_new = [xnew; unew]
    T     = new_traj.T
    m     = size(new_traj.K,1)
    kldiv = zeros(T)
    for t = 1:T
        μt    = μ_new[:,t]
        Σt    = Σ_new[:,:,t]
        Kp    = prev_traj.fx[:,:,t]
        Kn    = new_traj.fx[:,:,t]
        kp    = prev_traj.μu[:,t]
        kn    = new_traj.μu[:,t]
        Σp    = prev_traj.Σ[:,:,t]
        Σn    = new_traj.Σ[:,:,t]
        dim   = size(Σp,1)

        kldiv[t] = 1/2 * (trace(Σp\Σn) + (kp-kn)'(Σp\(kp-kn)) - dim + logdet(inv(Σp)) - logdet(inv(Σn)) )
        kldiv[t] = max(0,kldiv[t])
    end
    return sum(kldiv)
end


"""
Calculate the Q terms related to the KL-constraint.
Qtt is [Qxx Qxu; Qux Quu]
Qt is [Qx; Qu]
The Q terms calculated in the backwards pass should be added to these terms to produce the final Q terms.
This function should be called from within the backwards_pass function.
"""
function dkl(traj_new)
    m,n,T  = traj_new.m,traj_new.n,traj_new.T
    cx,cu,cxx,cuu,cxu = zeros(n,T),zeros(m,T),zeros(n,n,T),zeros(m,m,T),zeros(n,m,T)
    for t in 1:T
        K, k       = traj_new.fx[:,:,t], traj_new.μu[:,t]
        Σi         = inv(traj_new.Σ[:,:,t])
        cx[:,t],cu[:,t],cxx[:,:,t],cuu[:,:,t],cxu[:,:,t]  = dkli(Σi,K,k)
    end
    return cx,cu,cxx,cuu,cxu
end


function dkli(Σi,K,k)
    cx  = K'*Σi*k
    cu  = -Σi*k
    cxx = K'*Σi*K
    cuu = -Σi*k
    cxu = -K'*Σi
    cx,cu,cxx,cuu,cxu
end

function KLmv(Σi,K,k)
    M =
    [K'*Σi*K  -K'*Σi;
    -Σi*K    Σi ]
    v = [K'*Σi*k  -Σi*k]
    M,v
end


# TODO: Calculate Σ in the forwards pass, requires covariance of forward dynamics model. Is this is given by the Pkn matrix from the Kalman model?
# TODO: implement dual gradient descent in iLQG
