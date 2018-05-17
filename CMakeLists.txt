cmake_minimum_required(VERSION 2.8.3)
project(camera_model)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC -fopenmp")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    )


find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

#set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Eigen library path: ${EIGEN3_INCLUDE_DIR}")

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES camera_model
    CATKIN_DEPENDS roscpp std_msgs
#    DEPENDS system_lib
    )

include_directories(
    ${catkin_INCLUDE_DIRS}
     )

include_directories("include")


add_executable(Calibration 
    src/intrinsic_calib.cc
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/camera_models/PolyFisheyeCamera.cpp
    src/camera_models/SplineCamera.cpp
    src/camera_models/FovCamera.cpp
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/code_utils/math_utils/Polynomial.cpp
    src/code_utils/cv_utils.cc
    src/gpl/EigenQuaternionParameterization.cc
    )


add_library(camera_model STATIC
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/camera_models/PolyFisheyeCamera.cpp
    src/camera_models/SplineCamera.cpp
    src/camera_models/FovCamera.cpp
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/code_utils/math_utils/Polynomial.cpp
    src/gpl/EigenQuaternionParameterization.cc)

target_link_libraries(Calibration
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
     dw
    )
target_link_libraries(camera_model
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES} )
