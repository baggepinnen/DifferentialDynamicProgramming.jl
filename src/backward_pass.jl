vectens(a,b) = permutedims(sum(a.*b,1), [3 2 1])

macro setupQTIC()
    quote
        m          = size(u,1)
        n,N        = size(fx,1,3)

        cx         = reshape(cx, (n, N))
        cu         = reshape(cu, (m, N))
        cxx        = reshape(cxx, (n, n))
        cxu        = reshape(cxu, (n, m))
        cuu        = reshape(cuu, (m, m))

        k          = zeros(m,N)
        K          = zeros(m,n,N)
        Vx         = zeros(n,N)
        Vxx        = zeros(n,n,N)
        Quu        = Array{T}(m,m,N)
        Quui       = Array{T}(m,m,N)
        dV         = [0., 0.]

        Vx[:,N]    = cx[:,N]
        Vxx[:,:,N] = cxx
        Quu[:,:,N] = cuu
        if updateQuui
            Quui[:,:,N] = inv(Quu[:,:,N])
        end
        diverge    = 0
    end |> esc
end

macro end_backward_pass()
    quote
        if !isempty(kl_cost_terms)
            cxkl,cukl,cxxkl,cxukl,cuukl = kl_cost_terms
            η = isa(ηbracket,AbstractMatrix) ? ηbracket[2,t] : ηbracket[2]
            Qu = Qu / η + cukl
            Qux_reg = Qux_reg / η + cxukl
            QuuF = QuuF / η + cuukl
        end
        if isempty(lims) || lims[1,1] > lims[1,2]
            # debug("#  no control limits: Cholesky decomposition, check for non-PD")
            try
                R = chol(Hermitian(QuuF))
            catch
                diverge  = i
                return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx, dV
            end

            # debug("#  find control law")
            kK  = -R\(R'\[Qu Qux_reg])
            k_i = kK[:,1]
            K_i = kK[:,2:n+1]
        else
            # debug("#  solve Quadratic Program")
            lower = lims[:,1]-u[:,i]
            upper = lims[:,2]-u[:,i]
            result = 1
            # try
                k_i,result,R,free = boxQP(QuuF,Qu,lower,upper,k[:,min(i+1,N-1)])
            # catch
                # result = 0
            # end
            if result < 1
                diverge  = i
                return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx, dV
            end
            K_i  = zeros(m,n)
            if any(free)
                Lfree         = -R\(R'\Qux_reg[free,:])
                K_i[free,:]   = Lfree
            end
        end
        # debug("#  update cost-to-go approximation")
        # Qxx         = Hermitian(Qxx)
        dV         = dV + [k_i'Qu; .5*k_i'Quu[:,:,i]*k_i]
        Vx[:,i]    = Qx + K_i'Quu[:,:,i]*k_i + K_i'Qu + Qux'k_i
        Vxx[:,:,i] = Qxx + K_i'Quu[:,:,i]*K_i + K_i'Qux + Qux'K_i
        Vxx[:,:,i] = .5*(Vxx[:,:,i] + Vxx[:,:,i]')

        # debug("# save controls/gains")
        k[:,i]   = k_i
        K[:,:,i] = K_i
    end |> esc
end

function back_pass{T}(cx,cu,cxx::AbstractArray{T,3},cxu,cuu,fx::AbstractArray{T,3},fu,fxx,fxu,fuu,λ,regType,lims,x,u,updateQuui=false) # nonlinear time variant



    m,N        = size(u)
    n          = size(cx,1)
    cx         = reshape(cx, (n, N))
    cu         = reshape(cu, (m, N))
    cxx        = reshape(cxx, (n, n, N))
    cxu        = reshape(cxu, (n, m, N))
    cuu        = reshape(cuu, (m, m, N))
    k          = zeros(m,N)
    K          = zeros(m,n,N)
    Vx         = zeros(n,N)
    Vxx        = zeros(n,n,N)
    Quu        = Array{T}(m,m,N)
    Quui       = Array{T}(m,m,N)
    dV         = [0., 0.]
    Vx[:,N]    = cx[:,N]
    Vxx[:,:,N] = cxx[:,:,N]
    Quu[:,:,N] = cuu[:,:,N]
    if updateQuui
        Quui[:,:,N] = inv(Quu[:,:,N])
    end

    diverge  = 0
    for i = N-1:-1:1
        Qu  = cu[:,i]      + fu[:,:,i]'Vx[:,i+1]
        Qx  = cx[:,i]      + fx[:,:,i]'Vx[:,i+1]
        Qux = cxu[:,:,i]'  + fu[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        if !isempty(fxu)
            fxuVx = vectens(Vx[:,i+1],fxu[:,:,:,i])
            Qux   = Qux + fxuVx
        end

        Quu[:,:,i] = cuu[:,:,i]   + fu[:,:,i]'Vxx[:,:,i+1]*fu[:,:,i]
        if !isempty(fuu)
            fuuVx = vectens(Vx[:,i+1],fuu[:,:,:,i])
            Quu[:,:,i] .+= fuuVx
        end

        Qxx = cxx[:,:,i] + fx[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        isempty(fxx) || (Qxx .+= vectens(Vx[:,i+1],fxx[:,:,:,i]))
        Vxx_reg = Vxx[:,:,i+1] + (regType == 2 ? λ*eye(n) : 0)
        Qux_reg = cxu[:,:,i]'  + fu[:,:,i]'Vxx_reg*fx[:,:,i]
        isempty(fxu) || (Qux_reg .+= fxuVx)
        QuuF = cuu[:,:,i]  + fu[:,:,i]'Vxx_reg*fu[:,:,i] + (regType == 1 ? λ*eye(m) : 0)
        isempty(fuu) || (QuuF .+= fuuVx)

        @end_backward_pass
    end

    return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx,dV
end

# GaussianDist
# T::Int
# n::Int
# m::Int
# fx::Array{P,3}
# fu::Array{P,3}
# Σ::Array{P,3}
# μx::Array{P,2}
# μu::Array{P,2}


function back_pass{T}(cx,cu,cxx::AbstractArray{T,2},cxu,cuu,fx::AbstractArray{T,3},fu,fxx,fxu,fuu,λ,regType,lims,x,u,updateQuui=false) # quadratic timeinvariant cost, dynamics nonlinear time variant

    @setupQTIC

    for i = N-1:-1:1
        Qu  = cu[:,i]   + fu[:,:,i]'Vx[:,i+1]
        Qx  = cx[:,i]   + fx[:,:,i]'Vx[:,i+1]
        Qux = cxu' + fu[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        if !isempty(fxu)
            fxuVx = vectens(Vx[:,i+1],fxu[:,:,:,i])
            Qux   = Qux + fxuVx
        end

        Quu[:,:,i] = cuu + fu[:,:,i]'Vxx[:,:,i+1]*fu[:,:,i]
        if !isempty(fuu)
            fuuVx = vectens(Vx[:,i+1],fuu[:,:,:,i])
            Quu[:,:,i]   = Quu[:,:,i] + fuuVx
        end
        updateQuui && (Quui[:,:,i] = inv(Quu[:,:,i]))
        Qxx = cxx  + fx[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        isempty(fxx) || (Qxx .+= vectens(Vx[:,i+1],fxx[:,:,:,i]))
        Vxx_reg = Vxx[:,:,i+1] + (regType == 2 ? λ*eye(n) : 0)
        Qux_reg = cxu'   + fu[:,:,i]'Vxx_reg*fx[:,:,i]
        isempty(fxu) || (Qux_reg .+= fxuVx)
        QuuF = cuu  + fu[:,:,i]'Vxx_reg*fu[:,:,i] + (regType == 1 ? λ*eye(m) : 0)
        isempty(fuu) || (QuuF .+= fuuVx)

        @end_backward_pass
    end

    return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx,dV
end

function back_pass{T}(cx,cu,cxx::AbstractArray{T,2},cxu,cuu,fx::AbstractArray{T,3},fu,λ,regType,lims,x,u,updateQuui=false) # quadratic timeinvariant cost, linear time variant dynamics
    @setupQTIC
    for i = N-1:-1:1
        Qu         = cu[:,i] + fu[:,:,i]'Vx[:,i+1]
        Qx         = cx[:,i] + fx[:,:,i]'Vx[:,i+1]
        Qux        = cxu' + fu[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        Quu[:,:,i] = cuu + fu[:,:,i]'Vxx[:,:,i+1]*fu[:,:,i]
        Qxx        = cxx + fx[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        Vxx_reg    = Vxx[:,:,i+1] + (regType == 2 ? λ*eye(n) : 0)
        Qux_reg    = cxu' + fu[:,:,i]'Vxx_reg*fx[:,:,i]
        QuuF       = cuu + fu[:,:,i]'Vxx_reg*fu[:,:,i] + (regType == 1 ? λ*eye(m) : 0)
        updateQuui && (Quui[:,:,i] = inv(Quu[:,:,i]))
        @end_backward_pass
    end

    return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx,dV
end

function back_pass{T}(cx,cu,cxx::AbstractArray{T,3},cxu,cuu,fx::AbstractArray{T,3},fu,λ,regType,lims,x,u,updateQuui=false) # quadratic timeVariant cost, linear time variant dynamics
    m          = size(u,1)
    n,N        = size(fx,1,3)

    cx         = reshape(cx, (n, N))
    cu         = reshape(cu, (m, N))
    cxx        = reshape(cxx, (n, n, N))
    cxu        = reshape(cxu, (n, m, N))
    cuu        = reshape(cuu, (m, m, N))

    k          = zeros(m,N)
    K          = zeros(m,n,N)
    Vx         = zeros(n,N)
    Vxx        = zeros(n,n,N)
    Quu        = Array{T}(m,m,N)
    Quui       = Array{T}(m,m,N)
    dV         = [0., 0.]

    Vx[:,N]    = cx[:,N]
    Vxx[:,:,N] = cxx[:,:,end]
    Quu[:,:,N] = cuu[:,:,N]
    if updateQuui
        Quui[:,:,N] = inv(Quu[:,:,N])
    end
    diverge    = 0

    for i = N-1:-1:1
        Qu          = cu[:,i] + fu[:,:,i]'Vx[:,i+1]
        Qx          = cx[:,i] + fx[:,:,i]'Vx[:,i+1]
        Vxx_reg     = Vxx[:,:,i+1] + (regType == 2 ? λ*eye(n) : 0)
        Qux_reg     = cxu[:,:,i]' + fu[:,:,i]'Vxx_reg*fx[:,:,i]
        QuuF        = cuu[:,:,i]  + fu[:,:,i]'Vxx_reg*fu[:,:,i] + (regType == 1 ? λ*eye(m) : 0)
        Qux         = cxu[:,:,i]' + fu[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        Quu[:,:,i] .= cuu[:,:,i] .+ fu[:,:,i]'Vxx[:,:,i+1]*fu[:,:,i]
        Qxx         = cxx[:,:,i]  + fx[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        updateQuui && (Quui[:,:,i] = inv(Quu[:,:,i]))
        @end_backward_pass
    end

    return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx,dV
end

function back_pass{T}(cx,cu,cxx::AbstractArray{T,2},cxu,cuu,fx::AbstractMatrix{T},fu,λ,regType,lims,x,u,updateQuui=false) # cost quadratic and cost and LTI dynamics

    m,N = size(u)
    n   = size(fx,1)
    cx  = reshape(cx, (n, N))
    cu  = reshape(cu, (m, N))
    cxx = reshape(cxx, (n, n))
    cxu = reshape(cxu, (n, m))
    cuu = reshape(cuu, (m, m))
    k   = zeros(m,N)
    K   = zeros(m,n,N)
    Vx  = zeros(n,N)
    Vxx = zeros(n,n,N)
    Quu = Array{T}(m,m,N)
    Quui = Array{T}(m,m,N)
    dV  = [0., 0.]

    Vx[:,N]    = cx[:,N]
    Vxx[:,:,N] = cxx
    Quu[:,:,N] = cuu
    if updateQuui
        Quui[:,:,N] = inv(Quu[:,:,N])
    end

    diverge    = 0
    for i = N-1:-1:1
        Qu         = cu[:,i] + fu'Vx[:,i+1]
        Qx         = cx[:,i] + fx'Vx[:,i+1]
        Qux        = cxu' + fu'Vxx[:,:,i+1]*fx
        Quu[:,:,i] = cuu + fu'Vxx[:,:,i+1]*fu
        Qxx        = cxx + fx'Vxx[:,:,i+1]*fx
        Vxx_reg    = Vxx[:,:,i+1] + (regType == 2 ? λ*eye(n) : 0)
        Qux_reg    = cxu' + fu'Vxx_reg*fx
        QuuF       = cuu + fu'Vxx_reg*fu + (regType == 1 ? λ*eye(m) : 0)
        updateQuui && (Quui[:,:,i] = inv(Quu[:,:,i]))
        @end_backward_pass
    end

    return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx,dV
end




function graphics(x...)
    return 0
end
