option(BUILD_PERFORMANCE_TEST "Build performance test. Requires google benchmark library." ON)

if(BUILD_PERFORMANCE_TEST)
    find_package(benchmark REQUIRED)

    function (ndarray_add_perftest name)
        add_executable(${name} ${name}.cpp)
        target_link_libraries(${name} benchmark_main benchmark pthread)
    endfunction()

    ndarray_add_perftest(nd_array_iterator_perftest)
    ndarray_add_perftest(lazy_evaluation_perftest)
endif()
