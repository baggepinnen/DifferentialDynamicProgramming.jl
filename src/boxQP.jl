mutable struct QPTrace
    x
    xc
    value
    search
    clamped
    nfactor
end

"""
 Minimize `0.5*x'*H*x + x'*g`  s.t. lower<=x<=upper

  inputs:
     `H`            - positive definite matrix   (n * n)
     `g`            - bias vector                (n)
     `lower`        - lower bounds               (n)
     `upper`        - upper bounds               (n)

   optional inputs:
     `x0`           - initial state              (n)
     `options`      - see below                  (7)

  outputs:
     `x`            - solution                   (n)
     `result`       - result type (roughly, higher is better, see below)
     `Hfree`        - subspace cholesky factor   (n_free * n_free)
     `free`         - set of free dimensions     (n)
"""
function boxQP(H,g,lower,upper,x0::AbstractVector;
               maxIter        = 100,
               minGrad        = 1e-8,
               minRelImprove  = 1e-8,
               stepDec        = 0.6,
               minStep        = 1e-22,
               Armijo         = 0.1,
               print          = 0)
    #     maxIter        = 100        # maximum number of iterations
    #     minGrad        = 1e-8       # minimum norm of non-fixed gradient
    #     minRelImprove  = 1e-8       # minimum relative improvement
    #     stepDec        = 0.6        # factor for decreasing stepsize
    #     minStep        = 1e-22      # minimal stepsize for linesearch
    #     Armijo         = 0.1    	# Armijo parameter (fraction of linear improvement required)
    #     print          = 0 			# verbosity


    n        = size(H,1)
    clamped  = falses(n)
    free     = trues(n)
    oldvalue = 0.
    result   = 0
    gnorm    = 0.
    nfactor  = 0
    trace    = Array{QPTrace}(undef, maxIter)
    Hfree    = zeros(n,n)


    # debug("# initial state")
    x = clamp.(x0,lower,upper)
    LU = [lower upper]
    LU[.!isfinite.(LU)] .= NaN

    # debug("# initial objective value")
    value    = (x'g + 0.5x'H*x )[1]

    if print > 0
        @printf("==========\nStarting box-QP, dimension %-3d, initial value: %-12.3f\n",n, value)
    end

    # debug("# main loop")
    iter = 1
    while iter <= maxIter

        if result != 0
            break
        end

        # debug("# check relative improvement")
        if iter>1 && (oldvalue - value) < minRelImprove*abs(oldvalue)
            result = 4
            break
        end
        oldvalue = value

        # debug("# get gradient")
        grad     = g + H*x

        # debug("# find clamped dimensions")
        old_clamped = clamped
        clamped     = falses(n)
        #         clamped[(x[:,1] .== lower)&(grad[:,1].>0)]   = true
        #         clamped[(x[:,1] .== upper)&(grad[:,1].<0)]   = true
        for i = 1:n
            clamped[i]   = ((x[i,1] == lower[i])&&(grad[i,1]>0)) || ((x[i,1] == upper[i])&&(grad[i,1]<0))
        end
        free = .!clamped

        # debug("# check for all clamped")
        if all(clamped)
            result = 6
            break
        end

        # debug("# factorize if clamped has changed")
        if iter == 1
            factorize = true
        else
            factorize = any(old_clamped != clamped)
        end

        if factorize
            Hfree  = cholesky(H[free,free]).U  # was   (Hfree, indef)  = chol(H[free,free])
            #             if indef
            #                 result = -1
            #                 break
            #             end
            nfactor += 1
        end

        # debug("# check gradient norm")
        gnorm  = norm(grad[free])
        if gnorm < minGrad
            result = 5
            break
        end

        # debug("# get search direction")
        grad_clamped   = g  + H*(x.*clamped)
        search         = zeros(n,1)
        search[free]   = -Hfree\(Hfree'\grad_clamped[free]) - x[free]

        # debug("# check for descent direction")
        sdotg          = sum(search.*grad)
        if sdotg >= 0 # (should not happen)
            break
        end

        # debug("# armijo linesearch")
        step  = 1
        nstep = 0
        xc    = clamp.(x+step*search,lower,upper)
        vc    = (xc'*g + 0.5*xc'*H*xc)[1]
        while (vc - oldvalue)/(step*sdotg) < Armijo
            step  = step*stepDec
            nstep += 1
            xc    = clamp.(x+step*search,lower,upper)
            vc    = (xc'*g + 0.5*xc'*H*xc)[1]
            if step<minStep
                result = 2
                break
            end
        end

        if print > 1
            @printf("iter %-3d  value % -9.5g |g| %-9.3g  reduction %-9.3g  linesearch %g^%-2d  n_clamped %d\n",
                    iter, vc, gnorm, oldvalue-vc, stepDec, nstep, sum(clamped))
        end

        trace[iter] = QPTrace(x,xc,value,search,clamped,nfactor )

        # debug("# accept candidate")
        x     = xc
        value = vc
        iter += 1

    end

    if iter == maxIter
        result = 1
    end


    results = ["Hessian is not positive definite",          # result = -1
               "No descent direction found",                # result = 0    SHOULD NOT OCCUR
               "Maximum main iterations exceeded",          # result = 1
               "Maximum line-search iterations exceeded",   # result = 2
               "No bounds, returning Newton point",         # result = 3
               "Improvement smaller than tolerance",        # result = 4
               "Gradient norm smaller than tolerance",      # result = 5
               "All dimensions are clamped"]                # result = 6

    if print > 0
        @printf("RESULT: %s.\niterations %d  gradient %-12.6g final value %-12.6g  factorizations %d\n",
                results[result+2], iter, gnorm, value, nfactor)
    end


    return x,result,Hfree,free,trace
end

function demoQP(;kwargs...)
    n 		= 500
    g 		= randn(n)
    H 		= randn(n,n)
    H 		= H*H'
    lower 	= -ones(n)
    upper 	=  ones(n)
    @time boxQP(H, g, lower, upper, randn(n);print=1, kwargs...)

end
