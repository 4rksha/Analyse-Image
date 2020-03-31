cmake_minimum_required(VERSION 2.8)
project(ImageAnalysis )
find_package( OpenCV REQUIRED )
add_custom_target(build-time-make-directory ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory bin)
add_executable( bin/ImageAnalysis src/main.cpp )
target_link_libraries( bin/ImageAnalysis ${OpenCV_LIBS} )
