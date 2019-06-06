choleskyvectens(a,b) = permutedims(sum(a.*b,1), [3 2 1])


getmatindex(x,i) = x[i]
getmatindex(x::AbstractArray{<:Number},i) = x # In this case the array is constatnt over time
getmatindex(x::AbstractVector{<:AbstractArray},i) = x[i] # Time varying array

macro matviews(ex)
    ex = prewalk(ex) do ex
        @capture(ex, a_[i_] = rhs_) && (return :(__protect__($a,$i) = $rhs))
        @capture(ex, a_[i_] += rhs_) && (return :(__protect__($a,$i) += $rhs))
        @capture(ex, a_[i_]) && (return :(getmatindex($a,$i)))
        ex
    end
    ex = prewalk(ex) do ex
        @capture(ex, __protect__(a_,i_) = rhs_) && (return :($a[$i] = $rhs))
        @capture(ex, __protect__(a_,i_) += rhs_) && (return :($a[$i] += $rhs))
        ex
    end
    esc(ex)
end

@macroexpand @matviews begin
    dV         = dV + [k_i'Qu; .5*k_i'Quu[i]*k_i]
    Vx[i]    = Qx + K_i'Quu[i]*k_i + K_i'Qu + Qux'k_i
    Qu  = cu[i]      + fu[i]'Vx[i+1]
    Qx  = cx[i]      + fx[i]'Vx[i+1]
    Qux = cxu[i]'  + fu[i]'Vxx[i+1]*fx[i]
    fxuVx = vectens(Vx[i+1],fxu[i])
    Qux   = Qux + fxuVx
    Quu[i] = cuu[i]   + fu[i]'Vxx[i+1]*fu[i]
    fuuVx = vectens(Vx[i+1],fuu[i])
    Quu[i] += fuuVx
end

Base.zeros(::Type{Matrix{T}}, n::Int) where T = [zeros(T,0,0) for _ in 1:n]
Base.zeros(::Type{Vector{T}}, n::Int) where T = [T[] for _ in 1:n]

function back_pass(cx,cu,cxx,cxu,cuu,fx,fu,fxx,fxu,fuu,λ,regType,lims,x,u) # nonlinear time variant
    m,N        = length(u[1]),length(u)
    length(cx) == N || throw(ArgumentError("cx must be the same length as u"))
    @matviews begin
        n      = length(cx[1])
        k      = zeros(typeof(cu[1]), N)
        K      = zeros(typeof(copy(fu[1]')), N)
        Vx     = zeros(typeof(cx[1]), N)
        Vxx    = zeros(typeof(fx[1]), N)
        Quu    = zeros(typeof(cuu[1]), N)
        Quui   = zeros(typeof(cuu[1]), N)
        dV     = @SVector [0., 0.]
        Vx[N]  = cx[N]
        Vxx[N] = cxx[N]
        Quu[N] = cuu[N]
        k[N]   = 0*cu[N]
        K[N]   = 0*fu[N]'
    end

    diverge  = 0
    for i = N-1:-1:1
        @matviews begin
            Qu  = cu[i]      + fu[i]'Vx[i+1]
            Qx  = cx[i]      + fx[i]'Vx[i+1]
            Qux = cxu[i]'  + fu[i]'Vxx[i+1]*fx[i]
            if !(fxu === nothing)
                fxuVx = vectens(Vx[i+1],fxu[i])
                Qux   = Qux + fxuVx
            end

            Quu[i] = cuu[i]   + fu[i]'Vxx[i+1]*fu[i]
            if !(fuu === nothing)
                fuuVx = vectens(Vx[i+1],fuu[i])
                Quu[i] += fuuVx
            end

            Qxx = cxx[i] + fx[i]'Vxx[i+1]*fx[i]
            fxx === nothing || (Qxx .+= vectens(Vx[i+1],fxx[i]))
            Vxx_reg = Vxx[i+1] + (regType == 2 ? λ*I : 0*I)
            Qux_reg = cxu[i]'  + fu[i]'Vxx_reg*fx[i]
            fxu === nothing || (Qux_reg .+= fxuVx)
            QuuF = cuu[i]  + fu[i]'Vxx_reg*fu[i] + (regType == 1 ? λ*I : 0*I)
            fuu === nothing || (QuuF .+= fuuVx)
        end

        QuF = Qu
        if isempty(lims) || lims[1,1] > lims[1,2]
            # debug("#  no control limits: Cholesky decomposition, check for non-PD")
            local R
            try
                R = cholesky(Hermitian(QuuF))
            catch
                diverge  = i
                println("Failed chol")
                return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx, dV
            end
            # debug("#  find control law")
            k_i = -(R\QuF)
            K_i = -(R\Qux_reg)
        else
            # debug("#  solve Quadratic Program")
            lower = lims[:,1]-u[i]
            upper = lims[:,2]-u[i]
            local k_i,result,free
            try
                k_i,result,R,free = boxQP(QuuF,QuF,lower,upper,k[min(i+1,N-1)])
            catch
                result = 0
            end
            if result < 1
                diverge  = i
                println("Failed boxQP")
                return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx, dV
            end
            K_i  = similar(K[i])
            if any(free)
                Lfree         = -R\(R'\Qux_reg[free,:])
                K_i[free,:]   = Lfree
            end
        end
        # debug("#  update cost-to-go approximation")
        @matviews begin
            dV    += [k_i'Qu; .5*k_i'Quu[i]*k_i]
            Vx[i]  = Qx + K_i'Quu[i]*k_i + K_i'Qu + Qux'k_i
            Vxx[i] = Qxx + K_i'Quu[i]*K_i + K_i'Qux + Qux'K_i
            Vxx[i] = .5*(Vxx[i] + Vxx[i]')
        end

        # debug("# save controls/gains")
        k[i] = k_i
        K[i] = K_i
    end

    return diverge, GaussianPolicy(N,n,m,K,k,Quui,Quu), Vx, Vxx,dV
end



function graphics(x...)
    return 0
end

function back_pass_gps(cx,cu,cxx::AbstractArray{T,3},cxu,cuu, fx::AbstractArray{T,3},fu,lims,x,u,kl_cost_terms) where T # quadratic timeVariant cost, linear time variant dynamics
    m          = size(u,1)
    n,_,N      = size(fx)
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
    Quu        = Array{T}(undef,m,m,N)
    Quui       = Array{T}(undef,m,m,N)
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
                R = cholesky(Hermitian(Quu[:,:,i]))
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
