using DifferentialDynamicProgramming
using Test, Statistics, LinearAlgebra

@info("Compile time is high for this package, this is expected and is not an error.")
# write your own tests here

include("test_readme.jl")
demo_linear()
demo_linear_kl(kl_step=100)
demo_pendcart()

demoQP()

# include(Pkg.dir("GuidedPolicySearch","examples","bb.jl"))
