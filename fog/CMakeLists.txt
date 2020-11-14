cmake_minimum_required(VERSION 2.8.3)
project(fog)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge sensor_msgs std_msgs nav_msgs pcl_ros tf2_ros pcl_conversions message_filters)
find_package(OpenMP REQUIRED)
#find_package(PCL 1.9 REQUIRED)
#message(STATUS "***** PCL version: ${PCL_VERSION} *****")
if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})  # or if you use C: ${OpenMP_C_FLAGS}
  set(OpenMP_LIBS gomp)
endif()

#add_service_files(
#  FILES
#  comandoAquisitar.srv
#)

#generate_messages(
#  DEPENDENCIES
#  std_msgs
#)

include_directories(${catkin_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(../libraries/include)
include_directories(${OpenMP_INCLUDE_DIRS})

catkin_package(CATKIN_DEPENDS DEPENDS roscpp cv_bridge sensor_msgs std_msgs nav_msgs pcl_ros tf2_ros pcl_conversions message_filters)

add_executable(acc_space_node src/acc_space2.cpp ../libraries/src/registerobjectoptm.cpp ../libraries/src/processcloud.cpp)
target_compile_options(acc_space_node PRIVATE ${OpenMP_FLAGS})
add_dependencies(acc_space_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(acc_space_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OpenMP_LIBS})

#add_executable(planta_baixa_node src/planta_baixa.cpp ../libraries/src/registerobjectoptm.cpp ../libraries/src/processcloud.cpp)
#target_compile_options(planta_baixa_node PRIVATE ${OpenMP_FLAGS})
#add_dependencies(planta_baixa_node ${catkin_EXPORTED_TARGETS})
#target_link_libraries(planta_baixa_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OpenMP_LIBS})

add_executable(sfm_node src/sfm_horizontal.cpp ../libraries/src/sfm.cpp)
target_compile_options(sfm_node PRIVATE ${OpenMP_FLAGS})
add_dependencies(sfm_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(sfm_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OpenMP_LIBS})

#add_executable(imagem_virtual src/imagem_virtual.cpp ../libraries/src/registerobjectoptm.cpp ../libraries/src/processcloud.cpp)
#target_compile_options(imagem_virtual PRIVATE ${OpenMP_FLAGS})
#add_dependencies(imagem_virtual ${catkin_EXPORTED_TARGETS})
#target_link_libraries(imagem_virtual ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OpenMP_LIBS})

#add_executable(acc_obj_node src/acc_obj.cpp ../libraries/src/registerobjectoptm.cpp ../libraries/src/processcloud.cpp)
#target_compile_options(acc_obj_node PRIVATE ${OpenMP_FLAGS})
#add_dependencies(acc_obj_node ${catkin_EXPORTED_TARGETS})
#target_link_libraries(acc_obj_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OpenMP_LIBS})
