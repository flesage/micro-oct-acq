/*
 * FringeFFT.h
 *
 *  Created on: Dec 12, 2016
 *      Author: flesage
 */

#ifndef FRINGEFFT_H_
#define FRINGEFFT_H_

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cusparse.h>
#include <nppi.h>
#include "cuComplex.h"
#include <cublas.h>


class FringeFFT {
public:
	FringeFFT();
	virtual ~FringeFFT();
	void init( int nz, int nx);
    void set_disp_comp_vect(float* disp_comp_vector);
    void do_fft(float* in_fringe, cufftComplex* out_data);
    void interp_and_do_fft(float* in_fringe, cufftComplex* out_data);
    void compute_doppler(float line_period);
    void PutDopplerHPFilterOnGPU(float sigma, float lineperiod);
	void read_interp_matrix();
    void pre_compute_positions(int n_ang_pts, int n_radial_pts);
    void get_radial_img(Npp32f* fringe, Npp32f* interp_fringe);
private:
	cufftHandle plan;
    cublasHandle_t handle;
	cufftReal* d_fringe;
    cufftReal* d_interpfringe;
	cufftComplex* d_signal;
    cuFloatComplex* d_hann_dispcomp;
    cuFloatComplex* d_mean_fringe;
    float* d_ones;
    float* d_hp_filter;

	int p_nz;
	int p_nx;
	// Below is for sparse interpolation in k space
	cusparseHandle_t handle;
	cusparseMatDescr_t descr;
	float *dCsrValA;
	int *dCsrRowPtrA;
	int *dCsrColIndA;
	int totalNnz;

    Npp32f* d_nointerp;
    Npp32f* pXmap;
    Npp32f* pYmap;
    Npp32f* pDst;
    int p_n_radial_pts;

};

#endif /* FRINGEFFT_H_ */