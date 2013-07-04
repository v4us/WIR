project(SIF01)
cmake_minimum_required(VERSION 2.6)

find_package( OpenCV REQUIRED )
find_package (Threads)

add_definitions("-D_POSIX_C_SOURCE=200112L")

# Adding dl
set (EXTRA_LIBS ${EXTRA_LIBS} dl)

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

set (SOURCES
    mongoose.c
    )

set(SRC_LIST
    SIF01.cpp
    WIR01.cpp
    WIR_OCR.cpp
#    WIRDeamon.cpp
)

set (DEAMON_SRC
     WIRDeamon.cpp
     WIR01.cpp
     WIR_OCR.cpp
)

# Compiling library
add_library (_mongoose ${SOURCES})
target_link_libraries (_mongoose ${EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT} )

add_executable(${PROJECT_NAME} ${SRC_LIST})
add_executable(WIRDeamon  ${DEAMON_SRC})
add_executable(WIRServer WIR01.cpp WIR_OCR.cpp WIRServer.cpp)
target_link_libraries(WIRServer ${OpenCV_LIBS} _mongoose )
target_link_libraries(SIF01 ${OpenCV_LIBS} )
target_link_libraries(WIRDeamon ${OpenCV_LIBS} _mongoose )
#target_link_libraries(${PROJECT_NAME} rt dl m)

