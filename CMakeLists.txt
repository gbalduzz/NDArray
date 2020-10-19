cmake_minimum_required(VERSION 3.17)
project(ndarray)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/NumCpp)
include_directories(/home/giovanni/anaconda3/include)

option(BUILD_TESTS ON)
if(BUILD_TESTS)
    find_package(GTest)
    if (GTEST_FOUND)
        enable_testing()
        add_subdirectory(test)
    else()
        message("GTEST library not found.")
    endif()
endif()

install(DIRECTORY ndarray DESTINATION ./)
