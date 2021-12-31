
cmake_minimum_required(VERSION 3.9)

project(myapp VERSION 1.0.0)

# Ensure libmylib.so exists at the following path
find_library(MYLIB NAMES mylib PATHS "/home/samuel/dev/wise/load_library/lib/build")

add_executable(myapp main.cpp)

target_include_directories(myapp PUBLIC "../lib")
target_link_libraries(myapp PRIVATE ${MYLIB} ${CMAKE_DL_LIBS})
