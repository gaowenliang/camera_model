

#Main version:
##Source Tree:

* Refine from [CamOdoCal](https://github.com/hengli/camera_model)
* Refined by YZF [dvorak0](https://github.com/dvorak0)
* Modified by LIU Tiambo [GroundMelon](https://github.com/groundmelon)
* Modified by GWL [gaowenliang](https://github.com/gaowenliang)

## Acknowledgements ##

The origin author, Lionel Heng.

## Support Camera Models:

>Pinhole Camera

>Cata Camera

>Equidistant Camera

>Scaramuzza Camera Model

>Polynomial Fisheye Camera

This is my camera model

>Fov Camera

#Install

To use this package, need:

* [Eigen3](http://eigen.tuxfamily.org/)
* [ROS](http://wiki.ros.org/), almost use indigo version.
* [Ceres Solver](http://ceres-solver.org)

# Calibration:

Use [intrinsic_calib.cc](https://github.com/dvorak0/camera_model/blob/master/src/intrinsic_calib.cc) to calibrate your camera.
The template is like [fisheye_calibration.sh](https://github.com/gaowenliang/camera_model/blob/master/calibrate_template/fisheye_calibration.sh):

>  ./Calibration --camera-name mycamera --input mycameara_images/ -p IMG -e png -w 11 -h 8 --size 70 --camera-model myfisheye --opencv true


# USE:
Two main files for you to use camera model: [Camera.h](https://github.com/dvorak0/camera_model/blob/master/include/camera_model/camera_models/Camera.h) and [CameraFactory.h](https://github.com/gaowenliang/camera_model/blob/master/include/camera_model/camera_models/CameraFactory.h).
##1.load in the camera model calibration file
Use function in [CameraFactory.h](https://github.com/gaowenliang/camera_model/blob/master/include/camera_model/camera_models/CameraFactory.h) to load in the camra calibration file:

```c++
#include <camera_model/camera_models/CameraFactory.h>

camera_model::CameraPtr cam;

void loadCameraFile(std::string camera_model_file)
{
    cam = camera_model::CameraFactory::instance()->generateCameraFromYamlFile(camera_model_file);
}
```

##2.projection and back-projection point
See [Camera.h](https://github.com/dvorak0/camera_model/blob/master/include/camera_model/camera_models/Camera.h) for general interface:

Projection (3D ---> 2D) function:
[spaceToPlane](https://github.com/gaowenliang/camera_model/blob/master/calibrate_template/fisheye_calibration.sh) :Projects 3D points to the image plane (Pi function)

```c++
#include <camera_model/camera_models/CameraFactory.h>

camera_model::CameraPtr cam;

void loadCameraFile(std::string camera_model_file)
{
    cam = camera_model::CameraFactory::instance()->generateCameraFromYamlFile(camera_model_file);
}

void useProjection(Eigen::Vector3d P)
{
    Eigen::Vector2d p_dst;
    cam->spaceToPlane(P, p_dst);
}
```

Back Projection (2D ---> 3D) function:
[liftSphere](https://github.com/gaowenliang/camera_model/blob/master/calibrate_template/fisheye_calibration.sh):   Lift points from the image plane to the projective space.
```c++
#include <camera_model/camera_models/CameraFactory.h>

camera_model::CameraPtr cam;

void loadCameraFile(std::string camera_model_file)
{
    cam = camera_model::CameraFactory::instance()->generateCameraFromYamlFile(camera_model_file);
}

void useProjection(Eigen::Vector3d P)
{
    Eigen::Vector2d p_dst;
    cam->spaceToPlane(P, p_dst);
}

void useBackProjection(Eigen::Vector2d p)
{
    Eigen::Vector3d P_dst;
    cam->liftSphere(p, P_dst);
}
```
