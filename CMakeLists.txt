cmake_minimum_required(VERSION 3.5)
project(ekf_node)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(ekf_node src/ekf_node.cpp)

ament_target_dependencies(ekf_node
  rclcpp
  geometry_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  Eigen3
)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include/)

install(TARGETS
  ekf_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()

