using DifferentialDynamicProgramming
using Base.Test

info("Compile time is high for this package, this is expected and is not an error.")
# write your own tests here

demo_linear_kl(kl_step=100)
demo_linear()

include("test_readme.jl")

demoQP()
