#include <cuda.h>
#include <builtin_types.h>
#include "fringe_norm.h"
#include <cufft.h>
#include <stdio.h>

extern "C"


__global__ void magCUDA(cuComplex *data, float *mag, int dsz){
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < dsz){
        mag[idx]   = logf(cuCabsf(data[idx])+0.000001f);
	}
}

__global__ void phase_adjacentCUDA(int nx, int nz, cuComplex* c_filt_image, float* phase)
{
	int row  = threadIdx.x + blockDim.x * blockIdx.x;
	int column     = threadIdx.y + blockDim.y * blockIdx.y;
	if(column < (nx-1) && row < nz)
	{
		int index1 = column * nz + row;
		int index2 = (column+1) * nz + row;
		cuComplex tmp;
		tmp.x=(c_filt_image[index1].x*c_filt_image[index2].x+c_filt_image[index1].y*c_filt_image[index2].y);
		tmp.y=(-c_filt_image[index1].x*c_filt_image[index2].y+c_filt_image[index1].y*c_filt_image[index2].x);
		phase[index1]=atan(tmp.y/tmp.x);
	}
}

__global__ void complex_convolveCUDA(int nx, int nz, int n_hpf_pts, cuComplex* c_image, float* kernel, cuComplex* c_filt_image)
{
	int row  = threadIdx.x + blockDim.x * blockIdx.x;
	int column     = threadIdx.y + blockDim.y * blockIdx.y;

	if(column < nx && row < nz)
	{
		int index = column * nz + row;


		int k = n_hpf_pts / 2;

		cuComplex conv_Val;
		conv_Val.x = 0.;
		conv_Val.y = 0.;

		// Convolution along x
		int min_col = max(0,k-column);
		int max_col = min(n_hpf_pts, nx-1-column+k);
		for (int i = min_col; i < max_col; i++)
		{
			// Pixel
			int offset_pix = index + nz*(i-k);
			conv_Val.x += kernel[i] * c_image[offset_pix].x;
			conv_Val.y += kernel[i] * c_image[offset_pix].y;

		}
		// Set value of pixel
		c_filt_image[index].x = conv_Val.x;
		c_filt_image[index].y = conv_Val.y;
	}
}

__global__ void fringeNormCUDA(float* fringe, const float* avg_fringe, const float* hann_window, int nx, int nz)
{
	int row  = threadIdx.x + blockDim.x * blockIdx.x;
	int column     = threadIdx.y + blockDim.y * blockIdx.y;
	int index = column * nz + row;


	if(column < nx && row < nz)
	{
        fringe[index] = ((fringe[index]/(avg_fringe[row]+0.00001))-1.0)*hann_window[row];
	}
}

void fringeNorm(float* fringe, const float* avg_fringe, const float* hann_window, int nx, int nz)
{
	dim3 threadsPerBlock (512,2,1);
	dim3 blocksPerGrid (1,1,1);
    blocksPerGrid.x=(unsigned int) ceil(double(nz)/double(threadsPerBlock.x));
    blocksPerGrid.y=(unsigned int) ceil(double(nx)/double(threadsPerBlock.y));
	fringeNormCUDA<<<blocksPerGrid, threadsPerBlock>>>(fringe, avg_fringe, hann_window, nx, nz);
}

void mag(cuComplex* data, float* mag, int dsz)
{
	dim3 threadsPerBlock (1024,1,1);
	dim3 blocksPerGrid (1,1,1);
    blocksPerGrid.x=(unsigned int) ceil(dsz/1024.0);
	magCUDA<<<blocksPerGrid, threadsPerBlock>>>(data, mag, dsz);

}

void complex_convolve(int nx, int nz, int p_hpf_npts, cuComplex* c_image, float* kernel, cuComplex* c_filt_image)
{
	dim3 threadsPerBlock (1024,1,1);
	dim3 blocksPerGrid (1,1,1);
    blocksPerGrid.x=(unsigned int) ceil(double(nz)/double(threadsPerBlock.x));
    blocksPerGrid.y=(unsigned int) ceil(double(nx)/double(threadsPerBlock.y));
	complex_convolveCUDA<<<blocksPerGrid, threadsPerBlock>>>(nx,nz,p_hpf_npts, c_image, kernel, c_filt_image);
}

void phase_adjacent(int nx, int nz, cuComplex* c_filt_image, float* phase)
{
	dim3 threadsPerBlock (1024,1,1);
	dim3 blocksPerGrid (1,1,1);
    blocksPerGrid.x=(unsigned int) ceil(double(nz)/double(threadsPerBlock.x));
    blocksPerGrid.y=(unsigned int) ceil(double(nx)/double(threadsPerBlock.y));
	phase_adjacentCUDA<<<blocksPerGrid, threadsPerBlock>>>(nx,nz, c_filt_image, phase);
}
