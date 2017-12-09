#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

using namespace std;

void shape( Mat& m )
{
	cout << "(" << m.rows << "," << m.cols << ")" << endl;
}

void dt( clock_t begin, clock_t end )
{
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << elapsed_secs << endl;
}

int main( int argc, char** argv )
{
	clock_t begin;

	// Load the source image
	Mat src =
	        cv::imread(
	                "/home/zaqwes/Dropbox/Pictures/the_composer_by_stellarstatelogic-d5srz6u.png",
//	                "/home/zaqwes/Dropbox/Pictures/84afdc887c2e0bb6507cb8e79fca6aae.jpg",
	                0);
	shape(src);

	// Create a destination Mat object
	Size s = src.size();
	Mat dst((int) (s.height * 0.3), (int) (s.width * 0.3),
	        DataType<uint8_t>::type);
	Size d = dst.size();

	// Resize
	begin = clock();
	resize(src, dst, dst.size());
	dt(begin, clock());
	shape(dst);

	// Smooth
	begin = clock();
	medianBlur(dst, src, 3);
	dt(begin, clock());

	// Thresh.
	begin = clock();
	adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
	        35, -5);
	dt(begin, clock());

	// Postprocessing
	int dilation_size = 1;
	Mat element = getStructuringElement(MORPH_RECT,
	        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
	        Point(dilation_size, dilation_size));
	// Apply the dilation operation
	begin = clock();
	dilate(dst, src, element);
	dt(begin, clock());

	// Countours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	begin = clock();
	findContours(src, contours, hierarchy, CV_RETR_CCOMP,
	        CV_CHAIN_APPROX_TC89_L1);
	dt(begin, clock());
	cout << contours.size() << endl;

	// Draw
	namedWindow("Before", CV_WINDOW_AUTOSIZE);
	// display the source image
	imshow("Before", src);
	waitKey();
	return 0;
}
