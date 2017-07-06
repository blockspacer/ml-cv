#include <vector>
#include <iostream>
#include "bitset"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>

#include <stdint.h>

using namespace std;
using namespace cv;

typedef unsigned char uchar;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
		true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

__device__
inline int clamp(int x, int min_, int max_) {
	if (x < min_)
		x = min_;
	else if (x > max_)
		x = max_;
	return x;
}

__global__
void cencus(const uchar* in, uint32_t* out, int w, int h) {
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;

	int k1 = 3;
	int hk1 = k1 >> 1;

	if (gx >= w) {
		return;
	}
	if (gy >= h)
		return;

	int pos = gy * w + gx;
	if (pos >= w * h) {
		return;
	}

	uint32_t res = 0;
	uint32_t mask = 0x01 << (k1 * k1 - 1);

	uchar I = in[pos];
	for (int dx = -hk1; dx < hk1; ++dx) {
		for (int dy = -hk1; dy < hk1; ++dy) {
			int x = clamp(gx + dx, 0, w);
			int y = clamp(gy + dy, 0, h);

			uchar Ie = in[y * w + x];
			if (I < Ie) {
				res |= mask;
			} else {
				res &= ~(mask);
			}
			mask >>= 1;
		}
	}

	out[pos] = res;
}

__global__ void sbm_census(uchar* i0, uchar* i1, int w, int h, short* d_disp) {

}

int main(void) {

	Mat img = imread("halo_art.jpg", 0);

	int w = img.cols;
	int h = img.rows;
	int N = w * h;

	vector<uint32_t> out(N);
	uchar *d_x = 0;
	uint32_t *d_y = 0;

	gpuErrchk(cudaMalloc(&d_x, N * sizeof(uchar)));
	gpuErrchk(cudaMalloc(&d_y, N * sizeof(uint32_t)));

	gpuErrchk(
			cudaMemcpy(d_x, img.data, N * sizeof(uchar),
					cudaMemcpyHostToDevice));
	// cudaMemcpy(d_y, y, N * sizeof(uint32_t), cudaMemcpyHostToDevice);

	{
		const dim3 wg(w / 16, h / 16);
		const dim3 bs(16, 16);

		// measure
		// https://devblogs.nvidia.com/parallelforall/how-implement-performance-metrics-cuda-cc/
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		// Perform SAXPY on 1M elements

		cencus<<<wg, bs>>>(d_x, d_y, w, h);

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		cudaEventRecord(stop);

		gpuErrchk(
				cudaMemcpy(&out[0], d_y, N * sizeof(uint32_t),
						cudaMemcpyDeviceToHost));

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		cout << "elapsed:" << milliseconds / 1e3 << endl;
	}

	// https://stackoverflow.com/questions/14581806/can-not-use-cv-32uc1
	Mat A = Mat(h, w, CV_32S, &out[0]);
	// Mat_<uint32_t> A(w, h, )

	Mat B;
	A.convertTo(B, CV_8U);
	// imwrite("out.png", B-img);
	imwrite("out.png", B);

	gpuErrchk(cudaFree(d_x));
	gpuErrchk(cudaFree(d_y));
}
