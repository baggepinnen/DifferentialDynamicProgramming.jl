using DifferentialDynamicProgramming
using Test

@info("Compile time is high for this package, this is expected and is not an error.")
# write your own tests here
demoQP()
include(DifferentialDynamicProgramming.dir("src","demo_linear.jl"))
demo_linear()
