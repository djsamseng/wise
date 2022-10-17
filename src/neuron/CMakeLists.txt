

add_library(neuron STATIC
    neuron.cu
    neuron.h)

target_compile_features(neuron PUBLIC cxx_std_11)
set_target_properties(neuron PROPERTIES CUDA_SEPARABLE_COMPILATION ON)