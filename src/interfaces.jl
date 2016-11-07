

# Model interface ====================================
"""
Model interface, implement the following functions\n
see also `AbstractCost`, `ModelAndCost`
```
fit_model(::Type{AbstractModel}, batch::Batch)::AbstractModel

predict(model::AbstractModel, x, u)

predict(model::AbstractModel, batch::Batch)

function df(model::AbstractModel, x, u, I::UnitRange)
    return fx,fu,fxx,fxu,fuu
end
```
"""
abstract AbstractModel

"""
    fit_model(::Type{AbstractModel}, x,u,xnew)::AbstractModel
Fits a model to data
"""
function fit_model(::Type{AbstractModel}, x,u,xnew)::AbstractModel
    error("This function is not implemented for your type")
    return model
end

"""Predict the next state given the current state and action"""
function predict(model::AbstractModel, x, u)
    error("This function is not implemented for your type")
end

function predict(model::AbstractModel, batch::Batch)
    error("This function is not implemented for your type")
end

"""Get the linearized dynamics at `x`,`u`"""
function df(model::AbstractModel, x, u, I::UnitRange)
    error("This function is not implemented for your type")
    return fx,fu,fxx,fxu,fuu
end
# Model interface ====================================


# Cost interface ====================================
"""
Cost interface, implement the following functions\n
see also `AbstractModel`, `ModelAndCost`
```
function cost(::Type{AbstractCost}, x::AbstractVector, u)::Number

function cost(::Type{AbstractCost}, x::AbstractMatrix, u)::AbstractVector

function cost_final(::Type{AbstractCost}, x::AbstractVector)::Number

function dc(::Type{AbstractCost}, x, u)
    return cx,cu,cxx,cuu,cxu
end
```
"""
abstract AbstractCost

function cost(::Type{AbstractCost}, x::AbstractVector, u)::Number
    error("This function is not implemented for your type")
    return c
end

function cost(::Type{AbstractCost}, x::AbstractMatrix, u)::AbstractVector
    error("This function is not implemented for your type")
    return c
end

function cost_final(::Type{AbstractCost}, x::AbstractVector)::Number
    error("This function is not implemented for your type")
    return c
end

function dc(::Type{AbstractCost}, x, u)
    error("This function is not implemented for your type")
    return cx,cu,cxx,cuu,cxu
end
# Cost interface ====================================


"""
1. Define types that implement the interfaces `AbstractModel` and `AbstractCost`.
2. Create object modelcost = ModelAndCost(model, cost)
3. Run macro @define_modelcost_functions(modelcost). This macro defines the following functions
```
f(x, u, i)  = f(modelcost, x, u, i)
fT(x)       = fT(modelcost, x)
df(x, u, I) = df(modelcost, x, u, I)
```
see also `AbstractModel`, `AbstractCost`
"""
type ModelAndCost
    model::AbstractModel
    cost::AbstractCost
end

function f(modelcost::ModelAndCost, x, u, i)
    xnew = predict(modelcost.model, x, u, i)
    c    = cost(modelcost.cost, x, u)
    return xnew, cost
end

function fT(modelcost::ModelAndCost, x)
    c    = cost_final(modelcost.cost, x, u)
    return c
end

function df(modelcost::ModelAndCost, x, u, I)
    fx,fu,fxx,fxu,fuu = df(modelcost.model, x, u, I)
    cx,cu,cxx,cuu,cxu = dc(modelcost.cost, x, u)
    return fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu
end

"""
    define_modelcost_functions(modelcost)
This macro defines the following functions
```
f(x, u, i)  = f(modelcost, x, u, i)
fT(x)       = fT(modelcost, x)
df(x, u, I) = df(modelcost, x, u, I)
```
These functions can only be defined for one type of `ModelAndCost`. If you have several different `ModelAndCost`s, define your functions manually.
see also `ModelAndCost`, `AbstractModel`, `AbstractCost`
"""
macro define_modelcost_functions(modelcost)
    ex = quote
        f(x, u, i)  = DifferentialDynamicProgramming.f($modelcost, x, u, i)
        fT(x)       = DifferentialDynamicProgramming.fT($modelcost, x)
        df(x, u, I) = DifferentialDynamicProgramming.df($modelcost, x, u, I)
    end |> esc
    info("Defined:\nf(x, u, i)  = f($modelcost, x, u, i)\nfT(x) = fT($modelcost, x)\ndf(x, u, I) = df($modelcost, x, u, I)")
    return ex
end
