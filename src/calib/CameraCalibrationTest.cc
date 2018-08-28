#include "camera_model/calib/CameraCalibrationTest.h"

#include <algorithm>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camera_model/camera_models/CameraFactory.h"
#include "camera_model/camera_models/CostFunctionFactory.h"
#include "camera_model/gpl/EigenQuaternionParameterization.h"
#include "camera_model/gpl/EigenUtils.h"
#include "camera_model/sparse_graph/Transform.h"

#include "ceres/ceres.h"

namespace camera_model
{

CameraCalibrationTest::CameraCalibrationTest( )
: m_boardSize( cv::Size( 0, 0 ) )
, m_squareSize( 0.0f )
, m_verbose( false )
{
}

CameraCalibrationTest::CameraCalibrationTest( const std::string& cameraFile, //
                                              const cv::Size& boardSize,
                                              float squareSize )
: m_boardSize( boardSize )
, m_squareSize( squareSize )
, m_verbose( false )
{
    m_camera = CameraFactory::instance( )->generateCameraFromYamlFile( cameraFile );
}

void
CameraCalibrationTest::clear( void )
{
    m_imagePoints.clear( );
    m_scenePoints.clear( );
    m_imageGoodPoints.clear( );
    m_sceneGoodPoints.clear( );
}

void
CameraCalibrationTest::addImage( const cv::Mat image, const std::string name )
{
    cbImages.push_back( image );
    cbImageNames.push_back( name );
}

void
CameraCalibrationTest::addChessboardData( const std::vector< cv::Point2f >& corners )
{
    m_imagePoints.push_back( corners );

    std::vector< cv::Point3f > scenePointsInView;
    for ( int i = 0; i < m_boardSize.height; ++i )
    {
        for ( int j = 0; j < m_boardSize.width; ++j )
        {
            scenePointsInView.push_back( cv::Point3f( i * m_squareSize, j * m_squareSize, 0.0 ) );
        }
    }
    m_scenePoints.push_back( scenePointsInView );
}

bool
CameraCalibrationTest::calibrate( void )
{
    // compute intrinsic camera parameters and extrinsic parameters for each of the views
    std::vector< cv::Mat > rvecs;
    std::vector< cv::Mat > tvecs;

    rvecs.assign( m_scenePoints.size( ), cv::Mat( ) );
    tvecs.assign( m_scenePoints.size( ), cv::Mat( ) );

    // STEP 1: Estimate intrinsics
    std::cout << "[" << m_camera->cameraName( ) << "] "
              << "# INFO: "
              << "Initialization intrinsic parameters" << std::endl
              << "# INFO: " << m_camera->parametersToString( )
              << "-----------------------------------------------" << std::endl;

    // STEP 2: Estimate extrinsics
#pragma omp parallel for
    for ( size_t i = 0; i < m_scenePoints.size( ); ++i )
    {
        m_camera->estimateExtrinsics( m_scenePoints.at( i ),
                                      m_imagePoints.at( i ),
                                      rvecs.at( i ),
                                      tvecs.at( i ) );
    }

    // STEP 3: Optimizatoin
    CalibrationDataTest( m_camera, rvecs, tvecs, m_imagePoints, m_scenePoints );

    // Compute measurement covariance.
    std::vector< std::vector< cv::Point2f > > errVec( m_imagePoints.size( ) );
    Eigen::Vector2d errSum  = Eigen::Vector2d::Zero( );
    size_t errCount         = 0;
    int numOfOutlinerPoints = 0;
    int numOfInlinerPoints  = 0;
    std::vector< cv::Mat > rvecsGood;
    std::vector< cv::Mat > tvecsGood;

    for ( size_t i = 0; i < m_imagePoints.size( ); ++i )
    {
        std::vector< cv::Point2f > estImagePoints;
        m_camera->projectPoints( m_scenePoints.at( i ), rvecs.at( i ), tvecs.at( i ), estImagePoints );

        std::vector< cv::Point3f > oneSscenePoints;
        std::vector< cv::Point2f > oneImagePoints;
        for ( size_t j = 0; j < m_imagePoints.at( i ).size( ); ++j )
        {
            cv::Point2f pObs = m_imagePoints.at( i ).at( j );
            cv::Point2f pEst = estImagePoints.at( j );

            cv::Point2f err = pObs - pEst;

            if ( ( err.x * err.x + err.y * err.y ) < 1.5 )
            {
                oneImagePoints.push_back( m_imagePoints.at( i ).at( j ) );
                oneSscenePoints.push_back( m_scenePoints.at( i ).at( j ) );
                ++numOfInlinerPoints;
            }
            else
            {
                ++numOfOutlinerPoints;
            }

            errVec.at( i ).push_back( err );

            errSum += Eigen::Vector2d( err.x, err.y );
        }
        if ( oneImagePoints.size( ) >= 8 )
        {
            m_imageGoodPoints.push_back( oneImagePoints );
            m_sceneGoodPoints.push_back( oneSscenePoints );
            rvecsGood.push_back( rvecs.at( i ) );
            tvecsGood.push_back( tvecs.at( i ) );

            if ( m_verbose )
            {
                cv::Mat imageShow, imageGoodShow;
                cbImages.at( i ).copyTo( imageShow );
                cbImages.at( i ).copyTo( imageGoodShow );
                m_ImagesShow.push_back( imageShow );
                m_ImagesGoodShow.push_back( imageGoodShow );
                m_ImageNames.push_back( cbImageNames.at( i ) );
                m_imagePointsShow.push_back( m_imagePoints.at( i ) );
                m_scenePointsShow.push_back( m_scenePoints.at( i ) );
                m_imageGoodPointsShow.push_back( oneImagePoints );
                m_sceneGoodPointsShow.push_back( oneSscenePoints );
                m_cameraPoseRvecsShow.push_back( rvecs.at( i ) );
                m_cameraPoseTvecsShow.push_back( tvecs.at( i ) );
            }

            errCount += m_imagePoints.at( i ).size( );
        }
    }
    std::cout << "Total Points: " << numOfInlinerPoints + numOfOutlinerPoints << " , " << numOfInlinerPoints
              << " inlier, " << numOfOutlinerPoints << " outlier." << std::endl
              << "-----------------------------------------------" << std::endl;

    // STEP 3: Optimizatoin with good measurement point
    Eigen::Vector2d errMean = errSum / static_cast< double >( errCount );

    Eigen::Matrix2d measurementCovariance = Eigen::Matrix2d::Zero( );
    for ( size_t i = 0; i < errVec.size( ); ++i )
    {
        for ( size_t j = 0; j < errVec.at( i ).size( ); ++j )
        {
            cv::Point2f err = errVec.at( i ).at( j );

            double d0 = err.x - errMean( 0 );
            double d1 = err.y - errMean( 1 );

            measurementCovariance( 0, 0 ) += d0 * d0;
            measurementCovariance( 0, 1 ) += d0 * d1;
            measurementCovariance( 1, 1 ) += d1 * d1;
        }
    }
    measurementCovariance /= static_cast< double >( errCount );
    measurementCovariance( 1, 0 ) = measurementCovariance( 0, 1 );

    m_measurementCovariance = measurementCovariance;

    return true;
}

int
CameraCalibrationTest::sampleCount( void ) const
{
    return m_imagePoints.size( );
}

std::vector< std::vector< cv::Point2f > >&
CameraCalibrationTest::imagePoints( void )
{
    return m_imagePoints;
}

const std::vector< std::vector< cv::Point2f > >&
CameraCalibrationTest::imagePoints( void ) const
{
    return m_imagePoints;
}

std::vector< std::vector< cv::Point3f > >&
CameraCalibrationTest::scenePoints( void )
{
    return m_scenePoints;
}

const std::vector< std::vector< cv::Point3f > >&
CameraCalibrationTest::scenePoints( void ) const
{
    return m_scenePoints;
}

CameraPtr&
CameraCalibrationTest::camera( void )
{
    return m_camera;
}

const CameraConstPtr
CameraCalibrationTest::camera( void ) const
{
    return m_camera;
}

Eigen::Matrix2d&
CameraCalibrationTest::measurementCovariance( void )
{
    return m_measurementCovariance;
}

const Eigen::Matrix2d&
CameraCalibrationTest::measurementCovariance( void ) const
{
    return m_measurementCovariance;
}

cv::Mat&
CameraCalibrationTest::cameraPoses( void )
{
    return m_cameraPoses;
}

const cv::Mat&
CameraCalibrationTest::cameraPoses( void ) const
{
    return m_cameraPoses;
}

void
CameraCalibrationTest::drawResultsInitial( std::vector< cv::Mat >& images,
                                           std::vector< std::string >& imageNames,
                                           cv::Mat& DistributedImage ) const
{
    drawResults( images,
                 imageNames, //
                 DistributedImage,
                 m_cameraPoseRvecsShow,
                 m_cameraPoseTvecsShow,
                 m_imagePointsShow,
                 m_scenePointsShow );
}

void
CameraCalibrationTest::drawResultsFiltered( std::vector< cv::Mat >& images,
                                            std::vector< std::string >& imageNames,
                                            cv::Mat& DistributedImage ) const
{
    drawResults( images,
                 imageNames,
                 DistributedImage, //
                 m_cameraPoseRvecsShow,
                 m_cameraPoseTvecsShow,
                 m_imageGoodPointsShow,
                 m_sceneGoodPointsShow );
}

void
CameraCalibrationTest::drawResults( std::vector< cv::Mat >& images,
                                    std::vector< std::string >& imageNames,
                                    const cv::Mat& DistributedImage,
                                    const std::vector< cv::Mat > cameraPoseRs,
                                    const std::vector< cv::Mat > cameraPoseTs,
                                    const std::vector< std::vector< cv::Point2f > > imagePoints,
                                    const std::vector< std::vector< cv::Point3f > > scenePoints ) const
{
    int drawShiftBits  = 4;
    int drawMultiplier = 1 << drawShiftBits;

    cv::Scalar yellow( 0, 255, 255 );
    cv::Scalar green( 0, 255, 0 );
    cv::Scalar red( 0, 0, 255 );

    int r_show       = 5;
    double show_unit = 1;
    show_unit
    = sqrt( images.at( 0 ).rows * images.at( 0 ).rows + images.at( 0 ).cols * images.at( 0 ).cols ) / 900;
    if ( show_unit < 1.0 )
        show_unit = 1.0;
    r_show = 5 * show_unit;

    size_t i;
#pragma omp parallel for private( i )
    for ( i = 0; i < images.size( ); ++i )
    {
        cv::Mat& image = images.at( i );

        if ( image.channels( ) == 1 )
        {
            cv::cvtColor( image, image, CV_GRAY2RGB );
        }

        std::vector< cv::Point2f > estImagePoints;
        m_camera->projectPoints( scenePoints.at( i ), cameraPoseRs.at( i ), cameraPoseTs.at( i ), estImagePoints );

        float errorSum = 0.0f;
        float errorMax = std::numeric_limits< float >::min( );

        for ( size_t j = 0; j < imagePoints.at( i ).size( ); ++j )
        {
            cv::Point2f pObs = imagePoints.at( i ).at( j );
            cv::Point2f pEst = estImagePoints.at( j );

            // green points is the observed points
            cv::circle( image,
                        cv::Point( cvRound( pObs.x * drawMultiplier ), cvRound( pObs.y * drawMultiplier ) ),
                        r_show,
                        green,
                        show_unit * 2,
                        CV_AA,
                        drawShiftBits );

            // red points is the estimated points
            cv::circle( image,
                        cv::Point( cvRound( pEst.x * drawMultiplier ), cvRound( pEst.y * drawMultiplier ) ),
                        r_show,
                        red,
                        show_unit * 2,
                        CV_AA,
                        drawShiftBits );

            float error = cv::norm( pObs - pEst );

            // yellow points is the observed points
            cv::circle( DistributedImage,
                        cv::Point( cvRound( pObs.x * drawMultiplier ), cvRound( pObs.y * drawMultiplier ) ),
                        r_show,
                        yellow,
                        show_unit * 2,
                        CV_AA,
                        drawShiftBits );

            // Print each error of chessboard point
            //      std::cout << pObs.x<<" "<<pObs.y<< " "<<error<<std::endl;

            errorSum += error;
            if ( error > errorMax )
            {
                errorMax = error;
            }
        }

        std::ostringstream oss;
        oss << "Error: avg:" << errorSum / imagePoints.at( i ).size( ) << " max:" << errorMax;

        cv::putText( image,
                     oss.str( ),
                     cv::Point( show_unit * 10, image.rows - show_unit * 10 ),
                     cv::FONT_HERSHEY_COMPLEX,
                     show_unit / 2,
                     cv::Scalar( 0, 100, 255 ),
                     show_unit,
                     CV_AA );

        cv::putText( image,
                     imageNames.at( i ),
                     cv::Point( show_unit * 10, show_unit * 20 ),
                     cv::FONT_HERSHEY_COMPLEX,
                     show_unit / 2,
                     cv::Scalar( 0, 100, 255 ),
                     show_unit,
                     CV_AA );
    }
}

void
CameraCalibrationTest::writeParams( const std::string& filename ) const
{
    m_camera->writeParametersToYamlFile( filename );
}

bool
CameraCalibrationTest::writeChessboardData( const std::string& filename ) const
{
    std::ofstream ofs( filename.c_str( ), std::ios::out | std::ios::binary );
    if ( !ofs.is_open( ) )
    {
        return false;
    }

    writeData( ofs, m_boardSize.width );
    writeData( ofs, m_boardSize.height );
    writeData( ofs, m_squareSize );

    writeData( ofs, m_measurementCovariance( 0, 0 ) );
    writeData( ofs, m_measurementCovariance( 0, 1 ) );
    writeData( ofs, m_measurementCovariance( 1, 0 ) );
    writeData( ofs, m_measurementCovariance( 1, 1 ) );

    writeData( ofs, m_cameraPoses.rows );
    writeData( ofs, m_cameraPoses.cols );
    writeData( ofs, m_cameraPoses.type( ) );
    for ( int i = 0; i < m_cameraPoses.rows; ++i )
    {
        for ( int j = 0; j < m_cameraPoses.cols; ++j )
        {
            writeData( ofs, m_cameraPoses.at< double >( i, j ) );
        }
    }

    writeData( ofs, m_imagePoints.size( ) );
    for ( size_t i = 0; i < m_imagePoints.size( ); ++i )
    {
        writeData( ofs, m_imagePoints.at( i ).size( ) );
        for ( size_t j = 0; j < m_imagePoints.at( i ).size( ); ++j )
        {
            const cv::Point2f& ipt = m_imagePoints.at( i ).at( j );

            writeData( ofs, ipt.x );
            writeData( ofs, ipt.y );
        }
    }

    writeData( ofs, m_scenePoints.size( ) );
    for ( size_t i = 0; i < m_scenePoints.size( ); ++i )
    {
        writeData( ofs, m_scenePoints.at( i ).size( ) );
        for ( size_t j = 0; j < m_scenePoints.at( i ).size( ); ++j )
        {
            const cv::Point3f& spt = m_scenePoints.at( i ).at( j );

            writeData( ofs, spt.x );
            writeData( ofs, spt.y );
            writeData( ofs, spt.z );
        }
    }

    return true;
}

bool
CameraCalibrationTest::readChessboardData( const std::string& filename )
{
    std::ifstream ifs( filename.c_str( ), std::ios::in | std::ios::binary );
    if ( !ifs.is_open( ) )
    {
        return false;
    }

    readData( ifs, m_boardSize.width );
    readData( ifs, m_boardSize.height );
    readData( ifs, m_squareSize );

    readData( ifs, m_measurementCovariance( 0, 0 ) );
    readData( ifs, m_measurementCovariance( 0, 1 ) );
    readData( ifs, m_measurementCovariance( 1, 0 ) );
    readData( ifs, m_measurementCovariance( 1, 1 ) );

    int rows, cols, type;
    readData( ifs, rows );
    readData( ifs, cols );
    readData( ifs, type );
    m_cameraPoses = cv::Mat( rows, cols, type );

    for ( int i = 0; i < m_cameraPoses.rows; ++i )
    {
        for ( int j = 0; j < m_cameraPoses.cols; ++j )
        {
            readData( ifs, m_cameraPoses.at< double >( i, j ) );
        }
    }

    size_t nImagePointSets;
    readData( ifs, nImagePointSets );

    m_imagePoints.clear( );
    m_imagePoints.resize( nImagePointSets );
    for ( size_t i = 0; i < m_imagePoints.size( ); ++i )
    {
        size_t nImagePoints;
        readData( ifs, nImagePoints );
        m_imagePoints.at( i ).resize( nImagePoints );

        for ( size_t j = 0; j < m_imagePoints.at( i ).size( ); ++j )
        {
            cv::Point2f& ipt = m_imagePoints.at( i ).at( j );
            readData( ifs, ipt.x );
            readData( ifs, ipt.y );
        }
    }

    size_t nScenePointSets;
    readData( ifs, nScenePointSets );

    m_scenePoints.clear( );
    m_scenePoints.resize( nScenePointSets );
    for ( size_t i = 0; i < m_scenePoints.size( ); ++i )
    {
        size_t nScenePoints;
        readData( ifs, nScenePoints );
        m_scenePoints.at( i ).resize( nScenePoints );

        for ( size_t j = 0; j < m_scenePoints.at( i ).size( ); ++j )
        {
            cv::Point3f& spt = m_scenePoints.at( i ).at( j );
            readData( ifs, spt.x );
            readData( ifs, spt.y );
            readData( ifs, spt.z );
        }
    }

    return true;
}

void
CameraCalibrationTest::setVerbose( bool verbose )
{
    m_verbose = verbose;
}

bool
CameraCalibrationTest::calibrateHelper( CameraPtr& camera,
                                        std::vector< cv::Mat >& rvecs,
                                        std::vector< cv::Mat >& tvecs,
                                        std::vector< std::vector< cv::Point2f > >& imagePoints,
                                        std::vector< std::vector< cv::Point3f > >& scenePoints ) const
{
    rvecs.assign( scenePoints.size( ), cv::Mat( ) );
    tvecs.assign( scenePoints.size( ), cv::Mat( ) );

    // STEP 1: Estimate intrinsics
    camera->estimateIntrinsics( m_boardSize, scenePoints, imagePoints );
    std::cout << "[" << camera->cameraName( ) << "] "
              << "# INFO: "
              << "Initialization intrinsic parameters" << std::endl
              << "# INFO: " << camera->parametersToString( )
              << "-----------------------------------------------" << std::endl;

    // STEP 2: Estimate extrinsics
    for ( size_t i = 0; i < scenePoints.size( ); ++i )
    {
        camera->estimateExtrinsics( scenePoints.at( i ),
                                    imagePoints.at( i ),
                                    rvecs.at( i ),
                                    tvecs.at( i ) );
    }
    return true;
}

double
CameraCalibrationTest::reprojectionError( const std::vector< std::vector< cv::Point3f > >& objectPoints,
                                          const std::vector< std::vector< cv::Point2f > >& imagePoints,
                                          const std::vector< cv::Mat >& rvecs,
                                          const std::vector< cv::Mat >& tvecs,
                                          cv::OutputArray _perViewErrors ) const
{
    int imageCount     = objectPoints.size( );
    size_t pointsSoFar = 0;
    double totalErr    = 0.0;

    bool computePerViewErrors = _perViewErrors.needed( );
    cv::Mat perViewErrors;
    if ( computePerViewErrors )
    {
        _perViewErrors.create( imageCount, 1, CV_64F );
        perViewErrors = _perViewErrors.getMat( );
    }

    int i = 0;
    for ( i = 0; i < imageCount; ++i )
    {
        size_t pointCount = imagePoints.at( i ).size( );

        pointsSoFar += pointCount;

        std::vector< cv::Point2f > estImagePoints;
        m_camera->projectPoints( objectPoints.at( i ), rvecs.at( i ), tvecs.at( i ), estImagePoints );

        double err = 0.0;
        for ( size_t j = 0; j < imagePoints.at( i ).size( ); ++j )
        {
            err += cv::norm( imagePoints.at( i ).at( j ) - estImagePoints.at( j ) );
        }

        if ( computePerViewErrors )
        {
            perViewErrors.at< double >( i ) = err / pointCount;
        }

        std::cout << " checcboard image " << i << ", reprojection error "
                  << "\033[31;47;1m" << err / imagePoints.at( i ).size( ) << "\033[0m"
                  << "\n";

        totalErr += err;
    }

    return totalErr / pointsSoFar;
}

bool
CameraCalibrationTest::CalibrationDataTest( CameraPtr& camera,
                                            std::vector< cv::Mat >& rvecs,
                                            std::vector< cv::Mat >& tvecs,
                                            std::vector< std::vector< cv::Point2f > >& imagePoints,
                                            std::vector< std::vector< cv::Point3f > >& scenePoints ) const
{

    optimize( camera, rvecs, tvecs, imagePoints, scenePoints );

    if ( m_verbose )
    {
        double err = reprojectionError( scenePoints, imagePoints, rvecs, tvecs );
        double rms = camera->reprojectionRMSError( scenePoints, imagePoints, rvecs, tvecs );

        std::cout << "[" << camera->cameraName( ) << "] "
                  << "# INFO: Final reprojection error: "
                  << "\033[31;47;1m" << err << "\033[0m"
                  << " pixels" << std::endl;
        std::cout << "[" << camera->cameraName( ) << "] "
                  << "# INFO: RMS reprojection error: "
                  << "\033[31;47;1m" << rms << "\033[0m"
                  << " pixels" << std::endl;
        std::cout << "[" << camera->cameraName( ) << "] "
                  << "# INFO: " << camera->parametersToString( )
                  << "-----------------------------------------------" << std::endl;
    }

    return true;
}

void
CameraCalibrationTest::optimize( CameraPtr& camera,
                                 std::vector< cv::Mat >& rvecs,
                                 std::vector< cv::Mat >& tvecs,
                                 const std::vector< std::vector< cv::Point2f > >& imagePoints,
                                 const std::vector< std::vector< cv::Point3f > >& scenePoints ) const
{
    // Use ceres to do optimization
    ceres::Problem problem;

    std::vector< Transform, Eigen::aligned_allocator< Transform > > transformVec( rvecs.size( ) );
    for ( size_t i = 0; i < rvecs.size( ); ++i )
    {
        Eigen::Vector3d rvec;
        cv::cv2eigen( rvecs.at( i ), rvec );

        transformVec.at( i ).rotation( ) = Eigen::AngleAxisd( rvec.norm( ), rvec.normalized( ) );
        transformVec.at( i ).translation( ) << tvecs[i].at< double >( 0 ),
        tvecs[i].at< double >( 1 ), tvecs[i].at< double >( 2 );
    }

    // create residuals for each observation
    for ( size_t i = 0; i < imagePoints.size( ); ++i )
    {
        for ( size_t j = 0; j < imagePoints.at( i ).size( ); ++j )
        {
            const cv::Point3f& spt = scenePoints.at( i ).at( j );
            const cv::Point2f& ipt = imagePoints.at( i ).at( j );

            ceres::CostFunction* costFunction = CostFunctionFactory::instance( )->generateCostFunction(
            camera, Eigen::Vector3d( spt.x, spt.y, spt.z ), Eigen::Vector2d( ipt.x, ipt.y ), CAMERA_POSE );

            ceres::LossFunction* lossFunction = new ceres::CauchyLoss( 1.0 );
            problem.AddResidualBlock( costFunction,
                                      lossFunction,
                                      transformVec.at( i ).rotationData( ),
                                      transformVec.at( i ).translationData( ) );
        }

        ceres::LocalParameterization* quaternionParameterization = new EigenQuaternionParameterization;

        problem.SetParameterization( transformVec.at( i ).rotationData( ), quaternionParameterization );
    }

    ceres::Solver::Options options;
    options.max_num_iterations         = 1000;
    options.num_threads                = 4;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.logging_type               = ceres::SILENT;
    if ( m_verbose )
    {
        options.minimizer_progress_to_stdout = true;
    }

    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );

    std::cout << summary.FullReport( ) << std::endl;

    for ( size_t i = 0; i < rvecs.size( ); ++i )
    {
        Eigen::AngleAxisd aa( transformVec.at( i ).rotation( ) );

        Eigen::Vector3d rvec = aa.angle( ) * aa.axis( );
        cv::eigen2cv( rvec, rvecs.at( i ) );

        cv::Mat& tvec          = tvecs.at( i );
        tvec.at< double >( 0 ) = transformVec.at( i ).translation( )( 0 );
        tvec.at< double >( 1 ) = transformVec.at( i ).translation( )( 1 );
        tvec.at< double >( 2 ) = transformVec.at( i ).translation( )( 2 );
    }
}

template< typename T >
void
CameraCalibrationTest::readData( std::ifstream& ifs, T& data ) const
{
    char* buffer = new char[sizeof( T )];

    ifs.read( buffer, sizeof( T ) );

    data = *( reinterpret_cast< T* >( buffer ) );

    delete buffer;
}

template< typename T >
void
CameraCalibrationTest::writeData( std::ofstream& ofs, T data ) const
{
    char* pData = reinterpret_cast< char* >( &data );

    ofs.write( pData, sizeof( T ) );
}
}
