
"""
Compute KL divergence between new and previous trajectory
distributions.

μ_new: (n+m)×T, mean of new trajectory distribution (xnew, unew).
Σ_new: n×n×T , variance of new trajectory distribution.
"""
function kl_div(xnew,unew, Σ_new, new_traj::GaussianDist, prev_traj::GaussianDist)
    (isempty(new_traj) || isempty(prev_traj)) && (return 0)
    μ_new = [xnew; unew]
    T     = new_traj.T
    # m     = size(new_traj.fu,1)
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
        Σip   = inv(Σp + 1e-5*I) # TODO: I added some regularization here, should maybe be a hyper parameter?
        Σin   = inv(Σn + 1e-5*I)
        Mp,vp = KLmv(Σip,Kp,kp)
        Mn,vn = KLmv(Σin,Kn,kn)
        cp    = prev_traj.dV[2]
        cn    = new_traj.dV[2]

        kldiv[t] = -0.5μt'(Mn-Mp)*μt -  μt'(vn-vp) - cn + cp -0.5sum(Σt*(Mn-Mp)) -0.5logdet(Σn) + 0.5logdet(Σp)
        kldiv[t] = max(0,kldiv[t])
    end
    return sum(kldiv)
end

# TODO: jag höll på att fundera på hur KL-contraintet skall hanteras. Räknas det ut på rätt ställe och på rätt sätt?
function kl_div_wiki(xnew,unew, Σ_new, new_traj::GaussianDist, prev_traj::GaussianDist)
    μ_new = [xnew; unew]
    T,m     = new_traj.T, new_traj.m
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
        Σip   = inv(Σp + 1e-5*I) # TODO: I added some regularization here, should maybe be a hyper parameter?
        Σin   = inv(Σn + 1e-5*I)
        dim   = size(Σp,1)

        kldiv[t] = 1/2 * (trace(Σp\Σn) + (kp-kn)'(Σp\(kp-kn)) - dim + logdet(Σip) - logdet(Σin) )
        kldiv[t] = max(0,kldiv[t])
    end
    return sum(kldiv)
end


"""
Calculate the Q terms related to the KL-constraint.
Qtt is [Qxx Qxu; Qux Quu]
Qt is [Qx; Qu]
These terms should be added to the Q terms calculated in the backwards pass to produce the final Q terms.
This Function should be called from within the backwards_pass Function or just prior to it to adjust the cost derivative matrices.
"""
function dkl(traj_new)
    isempty(traj_new) && (return (0,0,0,0,0))
    m,n,T  = traj_new.m,traj_new.n,traj_new.T
    cx,cu,cxx,cuu,cxu = zeros(n,T),zeros(m,T),zeros(n,n,T),zeros(m,m,T),zeros(n,m,T)
    for t in 1:T
        K, k       = traj_new.fx[:,:,t], traj_new.μu[:,t]
        Σi         = inv(traj_new.Σ[:,:,t] + 1e-5*I)
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
    @show K', Σi, k
    v = [K'*Σi*k  -Σi*k]
    M,v
end


# TODO: Calculate Σ in the forwards pass, requires covariance of forward dynamics model. Is this is given by the Pkn matrix from the Kalman model?
# TODO: implement dual gradient descent in iLQG
