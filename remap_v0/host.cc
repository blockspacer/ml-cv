#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main() {
	// 1. Проецируем шахматку
	// 2. Добавляем дисторсию
	// 3. Снимаем дисторсию
	// 4. reproject??

	// read
	Mat img = imread("/mnt/d1/datasets/im0.png", 0);
	Size imageSize(img.rows, img.cols);

	cout << img.rows / 2 << " " << img.cols / 2 << endl;

	// generate chessboard, own coordinate system
	// https://stackoverflow.com/questions/212237/constants-and-compiler-optimization-in-c
	// http://www.gotw.ca/gotw/081.htm
	// https://stackoverflow.com/questions/3435026/can-const-correctness-improve-performance
	const auto quad_size = 7;
	auto step = 0.125;
	vector<Point3f> objectPoints;
	// 4F need for convertion
	for (auto x = 0; x < quad_size; ++x) {
		for (auto y = 0; y < quad_size; ++y) {
			Point3f p(x * step, y * step, 0);  // plain 2D object
			objectPoints.push_back(p);
		}
	}

	// internal
	vector<float> plain_K { 5299.313, 0, 1263.818, 0, 5299.313, 977.763, 0, 0, 1 };
	vector<float> plain_D { 0, 0, 0, 0 };

	// лучше какие-нибудь гомографии поделать
	Mat view, rview, map1, map2;
	Mat cameraMatrix(3, 3, CV_32FC1, &plain_K.front());
	Mat distCoeffs(1, 4, CV_32FC1, &plain_D.front());

	for (auto i = 0; i < 1; ++i) {
		// external
		vector<float> plain_T { 1, 0, 7 };
		Mat tvec(1, 3, CV_32FC1, &plain_T.front());

		Mat rvec;  // = Mat::zeros(1, 3, CV_32FC1);
		Mat R = Mat::eye(3, 3, CV_32FC1);
		Rodrigues(R, rvec);

		// Project chessboard
		vector<Point2f> imagePoints;
		projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs,
				imagePoints);

		for (auto p : imagePoints) {
			circle(img, p, 10, Scalar(255), 3);
			circle(img, p, 2, Scalar(0), 1);
		}

		imwrite("/tmp/proj.png", img);
	}

	// Stage N....

	auto P = cameraMatrix;
	//getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);

	if (0) {
		initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), P, imageSize,
		CV_32FC2, map1, map2);
		//cout << map1.size() << " " << map2.size() << endl;
		// how remap pixels from src
		// задано инвертное преобразование!!!
	}

	//undistortPoints

// solvePnP


// eval metirx

//// split to bins template

	{
		// https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html
		// просто по числу, возможно для итогового будет два селекта
		//
	}

	return 0;
}
