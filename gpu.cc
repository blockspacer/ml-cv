#include "opencv2/core/cuda.hpp"

#include <iostream>

using namespace std;

using namespace cv::cuda;
using namespace cv;

// g++ gpu.cc -L/opt/opencv-3.2.0_cuda-7.5/lib -lopencv_core

// https://github.com/opencv/opencv/blob/master/samples/gpu/morphology.cpp

int main()
{
	cuda::printCudaDeviceInfo(cuda::getDevice());

	// setGlDevice();

    Mat src1(640, 480, CV_8UC4, Scalar::all(0));
    Mat src2(640, 480, CV_8UC4, Scalar::all(0));

    // rectangle(src1, Rect(50, 50, 200, 200), Scalar(0, 0, 255, 128), 30);
    // rectangle(src2, Rect(100, 100, 200, 200), Scalar(255, 0, 0, 128), 30);

    GpuMat d_src1(src1);
    GpuMat d_src2(src2);

    GpuMat d_res;
}