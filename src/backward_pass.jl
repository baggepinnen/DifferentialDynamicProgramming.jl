vectens(a,b) = permutedims(sum(a.*b,1), [3 2 1])

macro setupQTIC()
    quote
        m          = size(u,1)
        n,N        = size(fx,1,3)

        @assert size(cx) == (n, N) "size(cx) should be (n, N)"
        @assert size(cu) == (m, N) "size(cu) should be (m, N)"
        @assert size(cxx) == (n, n) "size(cxx) should be(n, n) "
        @assert size(cxu) == (n, m) "size(cxu) should be(n, m) "

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
        diverge    = 0
    end |> esc
end

macro end_backward_pass()
    quote
        QuF = Qu
        if isempty(lims) || lims[1,1] > lims[1,2]
            # debug("#  no control limits: Cholesky decomposition, check for non-PD")
            local R
            try
                R = cholfact(Hermitian(QuuF))
            catch
                diverge  = i
                return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx, dV
            end
            # debug("#  find control law")
            k_i = -(R\QuF)
            K_i = -(R\Qux_reg)
        else
            # debug("#  solve Quadratic Program")
            lower = lims[:,1]-u[:,i]
            upper = lims[:,2]-u[:,i]
            local k_i,result,free
            try
                k_i,result,R,free = boxQP(QuuF,QuF,lower,upper,k[:,min(i+1,N-1)])
            catch
                result = 0
            end
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
        dV         = dV + [k_i'Qu; .5*k_i'Quu[:,:,i]*k_i]
        Vx[:,i]    = Qx + K_i'Quu[:,:,i]*k_i + K_i'Qu + Qux'k_i
        Vxx[:,:,i] = Qxx + K_i'Quu[:,:,i]*K_i + K_i'Qux + Qux'K_i
        Vxx[:,:,i] = .5*(Vxx[:,:,i] + Vxx[:,:,i]')

        # debug("# save controls/gains")
        k[:,i]   = k_i
        K[:,:,i] = K_i

    end |> esc
end

function back_pass{T}(cx,cu,cxx::AbstractArray{T,3},cxu,cuu,fx::AbstractArray{T,3},fu,fxx,fxu,fuu,λ,regType,lims,x,u) # nonlinear time variant

    m,N        = size(u)
    n          = size(cx,1)
    @assert size(cx) == (n, N)
    @assert size(cu) == (m, N)
    @assert size(cxx) == (n, n, N)
    @assert size(cxu) == (n, m, N)
    @assert size(cuu) == (m, m, N)
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


function back_pass{T}(cx,cu,cxx::AbstractArray{T,2},cxu,cuu,fx::AbstractArray{T,3},fu,fxx,fxu,fuu,λ,regType,lims,x,u) # quadratic timeinvariant cost, dynamics nonlinear time variant

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

function back_pass{T}(cx,cu,cxx::AbstractArray{T,2},cxu,cuu,fx::AbstractArray{T,3},fu,λ,regType,lims,x,u) # quadratic timeinvariant cost, linear time variant dynamics
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
        @end_backward_pass
    end

    return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx,dV
end

function back_pass{T}(cx,cu,cxx::AbstractArray{T,3},cxu,cuu,fx::AbstractArray{T,3},fu,λ,regType,lims,x,u) # quadratic timeVariant cost, linear time variant dynamics
    m          = size(u,1)
    n,N        = size(fx,1,3)

    @assert size(cx) == (n, N)
    @assert size(cu) == (m, N)
    @assert size(cxx) == (n, n, N)
    @assert size(cxu) == (n, m, N)
    @assert size(cuu) == (m, m, N)

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
        @end_backward_pass
    end

    return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx,dV
end

function back_pass{T}(cx,cu,cxx::AbstractArray{T,2},cxu,cuu,fx::AbstractMatrix{T},fu,λ,regType,lims,x,u) # cost quadratic and cost and LTI dynamics

    m,N = size(u)
    n   = size(fx,1)
    @assert size(cx) == (n, N)
    @assert size(cu) == (m, N)
    @assert size(cxx) == (n, n)
    @assert size(cxu) == (n, m)
    @assert size(cuu) == (m, m)
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
        @end_backward_pass
    end

    return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx,dV
end


function graphics(x...)
    return 0
end

function back_pass_gps{T}(cx,cu,cxx::AbstractArray{T,3},cxu,cuu, fx::AbstractArray{T,3},fu,lims,x,u,kl_cost_terms) # quadratic timeVariant cost, linear time variant dynamics
    m          = size(u,1)
    n,N        = size(fx,1,3)
    ηbracket = kl_cost_terms[2]
    η = isa(ηbracket,AbstractMatrix) ? ηbracket[2,N] : ηbracket[2]
    cxkl,cukl,cxxkl,cxukl,cuukl = kl_cost_terms[1]

    @assert size(cx) == (n, N)
    @assert size(cu) == (m, N)
    @assert size(cxx) == (n, n, N)
    @assert size(cxu) == (n, m, N)
    @assert size(cuu) == (m, m, N)

    k          = zeros(m,N)
    K          = zeros(m,n,N)
    Vx         = zeros(n,N)
    Vxx        = zeros(n,n,N)
    Quu        = Array{T}(m,m,N)
    Quui       = Array{T}(m,m,N)
    dV         = [0., 0.]

    Vx[:,N]    = cx[:,N]
    Vxx[:,:,N] = cxx[:,:,end]
    Quu[:,:,N] = cuu[:,:,N]./ η .+ cuukl[:,:,N]
    Quui[:,:,N] = inv(Quu[:,:,N])

    diverge    = 0
    for i = N-1:-1:1
        Qu          = cu[:,i] + fu[:,:,i]'Vx[:,i+1]
        Qx          = cx[:,i] + fx[:,:,i]'Vx[:,i+1]
        Qux         = cxu[:,:,i]' + fu[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]
        Quu[:,:,i] .= cuu[:,:,i] .+ fu[:,:,i]'Vxx[:,:,i+1]*fu[:,:,i]
        Qxx         = cxx[:,:,i]  + fx[:,:,i]'Vxx[:,:,i+1]*fx[:,:,i]

        ηbracket = kl_cost_terms[2]
        η = isa(ηbracket,AbstractMatrix) ? ηbracket[2,i] : ηbracket[2]
        Qu         = Qu ./ η + cukl[:,i]
        Qux        = Qux ./ η + cxukl[:,:,i]
        Quu[:,:,i] = Quu[:,:,i] ./ η + cuukl[:,:,i]
        Qx         = Qx ./ η + cxkl[:,i]
        Qxx        = Qxx ./ η + cxxkl[:,:,i]

        Quu[:,:,i] = 0.5*(Quu[:,:,i] + Quu[:,:,i]')

        if isempty(lims) || lims[1,1] > lims[1,2]
            # debug("#  no control limits: Cholesky decomposition, check for non-PD")
            local R
            try
                R = cholfact(Hermitian(Quu[:,:,i]))
            catch
                diverge  = i
                return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx, dV
            end

            # debug("#  find control law")
            k_i = -(R\Qu)
            K_i = -(R\Qux)
        else
            # debug("#  solve Quadratic Program")
            lower = lims[:,1]-u[:,i]
            upper = lims[:,2]-u[:,i]
            local k_i,result,free
            try
                k_i,result,R,free = boxQP(Quu[:,:,i],Qu,lower,upper,k[:,min(i+1,N-1)])
            catch
                result = 0
            end
            if result < 1
                diverge  = i
                return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx, dV
            end
            K_i  = zeros(m,n)
            if any(free)
                Lfree         = -R\(R'\Qux[free,:])
                K_i[free,:]   = Lfree
            end
        end
        # debug("#  update cost-to-go approximation")

        dV         = dV  + [k_i'Qu; .5*k_i'Quu[:,:,i]*k_i]
        Vx[:,i]    = Qx  + K_i'Quu[:,:,i]*k_i + K_i'Qu + Qux'k_i
        Vxx[:,:,i] = Qxx + K_i'Quu[:,:,i]*K_i + K_i'Qux + Qux'K_i
        Vxx[:,:,i] = .5*(Vxx[:,:,i] + Vxx[:,:,i]')

        # debug("# save controls/gains")
        k[:,i]   = k_i
        K[:,:,i] = K_i
        Quui[:,:,i] = inv(Quu[:,:,i])
    end

    return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx,dV
end
