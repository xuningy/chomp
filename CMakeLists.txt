cmake_minimum_required(VERSION 3.1)
project(chomp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")


find_package(catkin REQUIRED COMPONENTS
    bspline_opt
    voxblox_ros
		roscpp
    parameter_utils
)
find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
    INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIRS}
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS
      bspline_opt
      voxblox_ros
			roscpp
      parameter_utils
)

include_directories(include
      ${catkin_INCLUDE_DIRS}
)

link_directories(
      ${catkin_LIBRARY_DIRS}
)

add_definitions(
  ${catkin_DEFINITIONS}
)

add_library(${PROJECT_NAME}
  src/cost_map/CostMap.cpp
  src/chomp.cpp
)

add_dependencies(${PROJECT_NAME}
 	${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}
      ${catkin_LIBRARIES}
)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
      add_definitions(-DARMA_NO_DEBUG)
endif()


install(TARGETS ${PROJECT_NAME}
      ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
      LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
      RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY cmake/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.cmake"
)
