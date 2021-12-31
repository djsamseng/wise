
cmake_minimum_required(VERSION 3.9)

project(mylib VERSION 1.0.0)

find_package(OpenCV) #REQUIRED PATHS "/home/samuel/Downloads/opencv-4.x/build")
find_package(portaudio)

add_library(mylib SHARED
    mylib.cpp
    audio_capture.cpp
    desktop_capture.cpp)

set_target_properties(mylib PROPERTIES PUBLIC_HEADER "mylib.h")

target_include_directories(mylib PUBLIC
    "."
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(mylib PUBLIC ${OpenCV_LIBS} portaudio)

install(TARGETS mylib
    DESTINATION test
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include)

