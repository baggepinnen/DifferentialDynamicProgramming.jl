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
            dx = diff(xnew[:,i], x[:,i]) # TODO: verify if this is still reasonable
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

    sigmanew = cat(3,[eye(n) for i = 1:N]...) # TODO: this function should calculate the covariance matrix as well
    return xnew,unew,cnew,sigmanew
end
