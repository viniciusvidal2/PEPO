cmake_minimum_required(VERSION 2.8.3)
project(panoramica_esfera)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge sensor_msgs std_msgs image_transport nav_msgs pcl_ros tf2_ros tf tf_conversions camera_calibration_parsers camera_info_manager image_geometry laser_geometry)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})  # or if you use C: ${OpenMP_C_FLAGS}
  set(OpenMP_LIBS gomp)
endif()

include_directories(${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(../libraries/include)
include_directories(${OpenMP_INCLUDE_DIRS})

catkin_package(CATKIN_DEPENDS DEPENDS roscpp cv_bridge sensor_msgs std_msgs image_transport nav_msgs pcl_ros tf2_ros tf tf_conversions camera_calibration_parsers camera_info_manager image_geometry laser_geometry)

add_executable(panoramica src/projeta_esfera2.cpp)
target_compile_options(panoramica PRIVATE ${OpenMP_FLAGS})
add_dependencies(panoramica ${catkin_EXPORTED_TARGETS})
target_link_libraries(panoramica ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OpenMP_LIBS})
