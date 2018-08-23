"""
    xnew,unew,cnew,sigmanew = forward_pass(traj_new, x0,u,x,α,f,costfun,lims,diff)
# Arguments
- α: step size (αk is applied to old trajectory)
- diff: function to determine difference `diff(xnew[:,i], x[:,i])`
- f: forward dynamics `x(k+1)  = f(x(k), u(k), k)`
- `cnew = costfun(xnew, unew)`
"""
function forward_pass(traj_new, x0,u,x,α,f,costfun,lims,diff)
    n         = size(x0,1)
    m,N       = size(u)
    xnew      = Array{eltype(x0)}(n,N)
    xnew[:,1] = x0
    unew      = copy(u)
    cnew      = zeros(N)
    for i = 1:N
        if !isempty(traj_new)
            unew[:,i] .+= traj_new.k[:,i]*α
            dx = diff(xnew[:,i], x[:,i])
            unew[:,i] .+= traj_new.K[:,:,i]*dx
        end
        if !isempty(lims)
            unew[:,i] = clamp.(unew[:,i],lims[:,1], lims[:,2])
        end
        xnewi  = f(xnew[:,i], unew[:,i], i)
        if i < N
            xnew[:,i+1] = xnewi
        end
    end
    cnew = costfun(xnew, unew)

    return xnew,unew,cnew
end



function forward_covariance(model, x, u, traj)
    fx,fu,fxx,fxu,fuu = df(model, x, u)
    n        = size(fx,1)
    m        = size(fu,2)
    N        = size(fx,3)
    R1       = covariance(model,x,u) # Simple empirical prediction covariance
    Σ0       = R1 # TODO: I was lazy here
    ix = 1:n
    iu = n+1:n+m
    sigmanew = Array{Float64}(n+m,n+m,N)
    sigmanew[ix,ix,1] = Σ0
    for i = 1:N-1
        K,Σ = traj.K[:,:,i], traj.Σ[:,:,i]
        sigmanew[ix,ix,i+1] = fx[:,:,i]*sigmanew[ix,ix,i]*fx[:,:,i]' + R1 # Iterate dLyap forward
        sigmanew[iu,ix,i] = K*sigmanew[ix,ix,i]
        sigmanew[ix,iu,i] = sigmanew[ix,ix,i]*K'
        sigmanew[iu,iu,i] = K*sigmanew[ix,ix,i]*K' + Σ
    end
    sigmanew
end
