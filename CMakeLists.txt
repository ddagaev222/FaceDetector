cmake_minimum_required(VERSION 3.0)

project(face_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB H_FILES *.h)

file(GLOB SRC_FILES *.cpp)

add_executable(face_detector ${SRC_FILES} ${H_FILES})

target_link_libraries(face_detector ${OpenCV_LIBS})