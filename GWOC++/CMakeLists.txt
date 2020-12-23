cmake_minimum_required(VERSION 2.8.3)
project(GWO)

add_compile_options(-std=c++11)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message(STATUS OPENMP FOUND)
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
  SET(OpenMP_LIBS ${OpenMP_libomp_LIBRARY})
endif()
include_directories(${OpenMP_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
find_package(PCL 1.8.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

SET(HEADERS_BENCHMARK "${CMAKE_SOURCE_DIR}/include/benchmark")
include_directories("C:/Users/julia/Downloads/gwo-main/include")
include_directories(HEADERS)
include_directories(HEADERS_BENCHMARK)
SET(HEADERS
   include/argument.hpp  include/benchmark.hpp  include/benchmarkFactory.hpp   include/gwo.hpp  include/GWOException.hpp  include/utils.hpp include/benchmark/fob.hpp 
)
add_executable(main  ${HEADERS} src/main.cc src/argument.cc src/benchmark.cc src/benchmarkFactory.cc  src/gwo.cc src/GWOException.cc src/utils.cc benchmark/fob.cc)
target_compile_options(main PRIVATE ${OpenMP_FLAGS})
target_link_libraries(main    ${OpenCV_LIBS} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OpenMP_LIBS})


