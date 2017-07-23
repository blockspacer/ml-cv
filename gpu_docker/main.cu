#include <vector>
#include <iostream>
#include "bitset"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
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

__device__ int clamp_2d_position(int w, int h, int gx, int gy) {
	if (gx >= w) {
		return -1;
	}
	if (gy >= h)
		return -1;

	int pos = gy * w + gx;
	if (pos >= w * h) {
		return -1;
	}
	return pos;
}

__global__
void cencus_i3_xsobel5(const uchar* in, const uchar* in_xsobel, uint32_t* out,
		int w, int h) {
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;

	int pos = clamp_2d_position(w, h, gx, gy);
	if (pos < 0) {
		return;
	}

	int k1 = 3;
	int hk1 = k1 >> 1;

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

// [0, 32]
#define MAX_DISP 32
#define WS 11

#define SHIFT 4
#define FILTERED ((-1) << SHIFT)

__global__ void sbm_census(uint32_t* i0, uint32_t* i1, int w, int h,
		short* d_disp, int ws) {
	int gx_0 = blockIdx.x * blockDim.x + threadIdx.x;
	int gy_0 = blockIdx.y * blockDim.y + threadIdx.y;

	int pos_0 = clamp_2d_position(w, h, gx_0, gy_0);
	if (pos_0 < 0) {
		return;
	}

	d_disp[pos_0] = FILTERED;

	// fixme: можно кстати искать на меньшем диапазоне, а не выкидывать
	if (gx_0 < MAX_DISP) {
		return;
	}

	short shd[MAX_DISP];
	int hbs = ws >> 1;

	// SAD -> SHD (Sum of Hamming Distance (SHD))
	// WTA strategy

	// fixme: preload window - may be bad idia

	// (-max, 0]
	for (int d = 0; d < MAX_DISP; ++d) {
		int gx_1 = gx_0 - d;
		int gy_1 = gy_0;

		int shd_acc = 0;

		for (int x = -hbs; x < hbs; ++x) {
			for (int y = -hbs; y < hbs; ++y) {
				int ax0 = clamp(gx_0 + x, 0, w);
				int ay0 = clamp(gy_0 + y, 0, h);

				int ax1 = clamp(gx_1 + x, 0, w);
				int ay1 = clamp(gy_1 + y, 0, h);

				uint32_t I0 = i0[ay0 * w + ax0];
				uint32_t I1 = i1[ay1 * w + ax1];

				uint32_t dI = I0 ^ I1;

				// slow
				shd_acc += __popc(dI);
//				shd_acc += I0 + I1;

			}
		}

		shd[d] = shd_acc;
	}

	// minarg
	float minval = 1e6;
	int arg = -1;
	for (int d = 0; d < MAX_DISP; ++d) {
		if (minval > shd[d]) {
			arg = d;
			minval = shd[d];
		}
	}

	if (arg == -1) {
		d_disp[pos_0] = FILTERED;
		return;
	}

	// subpixel

	// uniqueless

	// text. thr.

	// store
	d_disp[pos_0] = arg << SHIFT;
}

void census(uchar* d_img, uchar* d_img_xsobel, uint32_t* d_img_census, int w,
		int h) {
	const dim3 wg(w / 16 + 1, h / 16 + 1);
	const dim3 bs(16, 16);
	// measure
	// https://devblogs.nvidia.com/parallelforall/how-implement-performance-metrics-cuda-cc/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	// Perform SAXPY on 1M elements

	cencus_i3_xsobel5<<<wg, bs>>>(d_img, d_img_xsobel, d_img_census, w, h);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "elapsed:" << milliseconds / 1e3 << endl;
}

struct gpu_matching_handle_t {

	gpu_matching_handle_t(int N, uchar* h_i0, uchar* h_i1, uchar* h_i0_xsobel,
			uchar* h_i1_xsobel) {
		d_i0 = 0;
		d_i0_census = 0;
		d_i1 = 0;
		d_i1_census = 0;

		// im0
		gpuErrchk(cudaMalloc(&d_i0, N * sizeof(uchar)));
		gpuErrchk(cudaMalloc(&d_i0_census, N * sizeof(uint32_t)));

		gpuErrchk(
				cudaMemcpy(d_i0, h_i0, N * sizeof(uchar),
						cudaMemcpyHostToDevice));

		// im1
		gpuErrchk(cudaMalloc(&d_i1, N * sizeof(uchar)));
		gpuErrchk(cudaMalloc(&d_i1_census, N * sizeof(uint32_t)));

		gpuErrchk(
				cudaMemcpy(d_i1, h_i1, N * sizeof(uchar),
						cudaMemcpyHostToDevice));

		// disp
		gpuErrchk(cudaMalloc(&d_disp_i16, N * sizeof(short)));
	}

	~gpu_matching_handle_t() {
		gpuErrchk(cudaFree(d_i0));
		gpuErrchk(cudaFree(d_i0_xsobel));
		gpuErrchk(cudaFree(d_i0_census));
		gpuErrchk(cudaFree(d_i1));
		gpuErrchk(cudaFree(d_i1_xsoble));
		gpuErrchk(cudaFree(d_i1_census));
		gpuErrchk(cudaFree(d_disp_i16));
	}

	uchar *d_i0;
	uchar *d_i0_xsobel;
	uint32_t *d_i0_census;
	uchar *d_i1;
	uchar *d_i1_xsoble;
	uint32_t *d_i1_census;

	short* d_disp_i16;
};

int main(void) {

	Mat im0 = imread("im0.png", 0);
	Mat im1 = imread("im1.png", 0);

	Mat dst0, dst1;
	double f = 0.2;
	cv::resize(im0, dst0, Size(0, 0), f, f);
	im0 = dst0;
	cv::resize(im1, dst1, Size(0, 0), f, f);
	im1 = dst1;

	int w = im0.cols;
	int h = im0.rows;
	int N = w * h;

	gpu_matching_handle_t mhandle(N, im0.data, im1.data, 0, 0);

	// census
	vector<uint32_t> h_i0_census(N);
	vector<uint32_t> h_i1_census(N);
	census(mhandle.d_i0, mhandle.d_i0_xsobel, mhandle.d_i0_census, w, h);

	gpuErrchk(
			cudaMemcpy(&h_i0_census[0], mhandle.d_i0_census,
					N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	// https://stackoverflow.com/questions/14581806/can-not-use-cv-32uc1
	Mat A = Mat(h, w, CV_32S, &h_i0_census[0]);

	Mat B;
	A.convertTo(B, CV_8U);
	imwrite("out0.png", B);

	census(mhandle.d_i1, mhandle.d_i0_xsobel, mhandle.d_i1_census, w, h);
	gpuErrchk(
			cudaMemcpy(&h_i1_census[0], mhandle.d_i1_census,
					N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	A = Mat(h, w, CV_32S, &h_i1_census[0]);

	A.convertTo(B, CV_8U);
	imwrite("out1.png", B);

	// matching
	{
		vector<short> h_disp(N, 0);
		const dim3 wg(w / 16 + 1, h / 16 + 1);
		const dim3 bs(16, 16);
		// measure
		// https://devblogs.nvidia.com/parallelforall/how-implement-performance-metrics-cuda-cc/
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		sbm_census<<<wg, bs>>>(mhandle.d_i0_census, mhandle.d_i1_census, w, h,
				mhandle.d_disp_i16, WS);

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		cout << "elapsed:" << milliseconds / 1e3 << endl;

		// store
		gpuErrchk(
				cudaMemcpy(&h_disp[0], mhandle.d_disp_i16, N * sizeof(short),
						cudaMemcpyDeviceToHost));

		A = Mat(h, w, CV_16S, &h_disp[0]);

		A.convertTo(B, CV_8U);
		imwrite("d_disp.png", B);
	}

}

