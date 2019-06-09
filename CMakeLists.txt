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

aux_source_directory(${PROJECT_SOURCE_DIR}/src/apriltag AprilTagSrc)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/apriltag_frontend AprilTagSrc2)

set(CAMERA_MODEL_FILES
    ${PROJECT_SOURCE_DIR}/src/camera_models/Camera.cc
    ${PROJECT_SOURCE_DIR}/src/camera_models/CostFunctionFactory.cc
    ${PROJECT_SOURCE_DIR}/src/camera_models/CameraFactory.cc
    ${PROJECT_SOURCE_DIR}/src/camera_models/PinholeCamera.cc
    ${PROJECT_SOURCE_DIR}/src/camera_models/PinholeFullCamera.cc
    ${PROJECT_SOURCE_DIR}/src/camera_models/CataCamera.cc
    ${PROJECT_SOURCE_DIR}/src/camera_models/EquidistantCamera.cc
    ${PROJECT_SOURCE_DIR}/src/camera_models/ScaramuzzaCamera.cc
    ${PROJECT_SOURCE_DIR}/src/camera_models/PolyFisheyeCamera.cpp
    ${PROJECT_SOURCE_DIR}/src/camera_models/SplineCamera.cpp
    ${PROJECT_SOURCE_DIR}/src/camera_models/FovCamera.cpp
    ${PROJECT_SOURCE_DIR}/src/sparse_graph/Transform.cc
    ${PROJECT_SOURCE_DIR}/src/gpl/gpl.cc
    ${PROJECT_SOURCE_DIR}/src/code_utils/math_utils/Polynomial.cpp
    ${PROJECT_SOURCE_DIR}/src/code_utils/cv_utils.cc
    ${PROJECT_SOURCE_DIR}/src/gpl/EigenQuaternionParameterization.cc
    ${AprilTagSrc}
    ${AprilTagSrc2}
    )

add_executable(Calibration 
    src/intrinsic_calib.cc
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    ${CAMERA_MODEL_FILES}
    )

add_executable(april_calib
    src/april_calib.cc
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    ${CAMERA_MODEL_FILES}
    )

 add_executable(Test
     src/test_calib.cc
     src/chessboard/Chessboard.cc
     src/calib/CameraCalibrationTest.cc
     ${CAMERA_MODEL_FILES}
     )

add_library(camera_model STATIC
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    ${CAMERA_MODEL_FILES}
    )

target_link_libraries(Calibration
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    dw    )
target_link_libraries(april_calib
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    dw    )
target_link_libraries(Test
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
     dw    )
target_link_libraries(camera_model
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES} )



