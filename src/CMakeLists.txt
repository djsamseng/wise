
add_subdirectory(neuron)

add_executable(wise-app main.cu)

target_compile_features(wise-app PUBLIC cxx_std_11)
set_property(TARGET wise-app
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(wise-app PRIVATE neuron -lcublas -lseccomp)
# set_property(TARGET wise-app PROPERTY CXX_STANDARD 20)