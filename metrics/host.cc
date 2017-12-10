#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

#include <cmath>

using namespace std;
using namespace cv;

void draw_points(Mat img, const vector<Point2f>& points, int c1, int c2, int r =
		15, bool ext = false) {
	for (auto p : points) {
		if (ext) {
			circle(img, p, r, Scalar(c1), 3);
		}
		circle(img, p, 5, Scalar(c2), 1); // center
	}
}

void draw_points(Mat img, const vector<Point2f>& points0,
		const vector<Point2f>& points1) {
	for (int i = 0; i < points0.size(); ++i) {
		arrowedLine(img, points0[i], points1[i], Scalar(128));
//		line(img, points0[i], points1[i], Scalar(255));
	}
}

class DatasetSplitter {
	// https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html
	// просто по числу, возможно для итогового будет два селекта
	// пока не берем в расчет вариабилити данных
	// https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
public:
	DatasetSplitter(int n) {
		this->n = n;
	}
	int n;

	int Rice() {
		// "It tends to overestimate the number of bins "
		//
		// "A good reason why the number of bins should be proportional to"
		return 2 * pow(n, 1.0 / 3);
	}

	int Sturges() {
		// "This estimator assumes normality of data and is
		// too conservative for larger, non-normal datasets. "
		//
		// " data and can perform poorly if n < 30"
		return log2(n) + 1;
	}

	int Sqrt() {
		return sqrt(n);
	}
};

// metrics
// inside chess ступеньками, но еще и по области картинки
class Parabola3P {
public:
	Parabola3P(Point2f p1, Point2f p2, Point2f p3) {
		ps = vector<Point2f> { p1, p2, p3 };
	}

	float getValue(float x) {
		// https://math.stackexchange.com/questions/889569/finding-a-parabola-from-three-points-algebraically
		// https://en.wikipedia.org/wiki/Lagrange_polynomial
		auto y1 = ps[0].y;
		auto y2 = ps[1].y;
		auto y3 = ps[2].y;

		auto x1 = ps[0].x;
		auto x2 = ps[1].x;
		auto x3 = ps[2].x;

		auto res = y1 * (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3))
				+ y2 * (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3))
				+ y3 * (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2));
		return res;
	}

	vector<Point2f> ps;
};

// Хочется не искать шахматки заново для проверки линейности
int main() {
	// 1. Проецируем шахматку
	// 2. Добавляем дисторсию
	// 3. Снимаем дисторсию
	// 4. reproject??

	// read
	Mat img = imread("/mnt/d1/datasets/im0.png", 0);
	Size imageSize(img.rows, img.cols);
	int h = img.rows;
	int w = img.cols;

	cout << img.rows / 2 << " " << img.cols / 2 << endl;

	// generate chessboard, own coordinate system
	// https://stackoverflow.com/questions/212237/constants-and-compiler-optimization-in-c
	// http://www.gotw.ca/gotw/081.htm
	// https://stackoverflow.com/questions/3435026/can-const-correctness-improve-performance
	const auto quad_size = 11;	//7;
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
	vector<float> plain_D_in { 3, 3, 0, 0 };
	vector<float> plain_D_out { -plain_D_in[0], -plain_D_in[1], 0, 0 };
	vector<float> d_real = plain_D_out;
	d_real[0] += 0.3;
	d_real[1] += 0.3;

	Mat cameraMatrix(3, 3, CV_32FC1, &plain_K.front());
	Mat distCoeffs_in(1, 4, CV_32FC1, &plain_D_in.front());
	Mat d_estimation(1, 4, CV_32FC1, &plain_D_out.front());

	Mat img_tmp0, img_tmp1;
	undistort(img, img_tmp0, cameraMatrix, distCoeffs_in);
	img = img_tmp0;

	vector<Point2f> imagePoints_in_last;
	vector<Point2f> imagePoints_out_norm;

	// external
	auto z = 5.0f + 0;
	z = 4;
	float x = 0;  //-1 + 0.5f * i;
	float y = 0;  // + 0.3f * i;
	vector<float> plain_T { x, y, z };
	Mat tvec(1, 3, CV_32FC1, &plain_T.front());

	Mat rvec;  // = Mat::zeros(1, 3, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32FC1);
	Rodrigues(R, rvec);
	rvec.at<float>(2, 0) = 1.5;
	rvec.at<float>(0, 0) = 1.5;

	// fixme: why out??? inverse transform? what for omnidir?
	projectPoints(objectPoints, rvec, tvec, cameraMatrix, d_estimation,
			imagePoints_in_last);
	draw_points(img, imagePoints_in_last, 128, 0, 15, true);

	//////////////////////////////////////
	// fixme: нет такого большого прироста в детектировании, для omnidir похоже нужно
	//   искать r and t и полностью репрожектить
	// Восстанавливаем
	undistort(img, img_tmp1, cameraMatrix, d_estimation);
	img = img_tmp1;

	// most important part
	undistortPoints(imagePoints_in_last, imagePoints_out_norm, cameraMatrix,
			d_real);

	float fxy = cameraMatrix.at<float>(0, 0);
	float cx = cameraMatrix.at<float>(0, 2);
	float cy = cameraMatrix.at<float>(1, 2);

	vector<Point2f> imagePoints_out_real;
	for (auto p : imagePoints_out_norm) {
		Point2f new_p(p.x * fxy + cx, p.y * fxy + cy);
		imagePoints_out_real.push_back(new_p);
	}

	// fixme: А как сделать solvePnP тут?
	draw_points(img, imagePoints_out_real, 255, 0, 20);

	// Stage N....

	//undistortPoints

	// solvePnP
	// fixme: how???
	Mat new_rvec, new_tvec;
	solvePnP(objectPoints, imagePoints_out_real, cameraMatrix, noArray(),
			new_rvec, new_tvec);
	vector<Point2f> imagePoints_reproj;
	projectPoints(objectPoints, new_rvec, new_tvec, cameraMatrix, noArray(),
			imagePoints_reproj);
	draw_points(img, imagePoints_out_real, imagePoints_reproj);
	draw_points(img, imagePoints_reproj, 128 + 32, 0, 30);

	// metric
	// по краям в три раза, от оптического центра
	// по шахматке ступенчато
	float rx = min(cx, w - cx);
	float ry = min(cy, h - cy);

	cout << "cx:" << cx << " cy:" << cy << " w:" << w << " h:" << h << endl;
	cout << "rx:" << rx << " ry:" << ry << endl;

	float min_r = min(rx, ry);

	float edge_w = 3;
	Parabola3P parabola( { -min_r, edge_w }, { 0, 0 }, { min_r, edge_w });
//	cout << parabola.getValue(-min_r) << endl;

	imwrite("/tmp/proj.png", img);

	// eval metirx

	//// split to bins template

	return 0;
}
