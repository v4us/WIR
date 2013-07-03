project(SIF01)
cmake_minimum_required(VERSION 2.6)

find_package( OpenCV REQUIRED )

add_definitions("-D_POSIX_C_SOURCE=200112L")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -pg")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE
        STRING "Choose the type of build, options are: Debug Release."
        FORCE)
endif()

set(SRC_LIST
    SIF01.cpp
    WIR01.cpp
    WIR_OCR.cpp
)

add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(SIF01 ${OpenCV_LIBS} )
#target_link_libraries(${PROJECT_NAME} rt dl m)

