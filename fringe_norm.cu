#include <cuda.h>
#include <builtin_types.h>
#include "fringe_norm.h"

extern "C"

__global__ void fringeNormCUDA(const float* fringe, const float* avg_fringe, const float* hann_window, int nx, int nz)
{
    int column  = threadIdx.x + blockDim.x * blockIdx.x;
    int row     = threadIdx.y + blockDim.y * blockIdx.y;
    int index = column * nz + row;

    if(column < nx && row < nz)
    {
        fringe[index] = ((fringe[index]/avg_fringe[row])-1.0)*hann_window[row];
    }
}

void fringeNorm(const float* fringe, const float* avg_fringe, const float* hann_window, int nx, int nz)
{
    dim3 threadsPerBlock (512, 512);
    dim3 blocksPerGrid (1,1);
    blocksPerGrid.x=ceil(double(nz)/double(threadsPerBlock.x));
    blocksPerGrid.y=ceil(double(nx)/double(threadsPerBlock.y));

    fringeNormCUDA<<<blocksPerGrid, threadsPerBlock>>>(fringe, avg_fringe, hann_window, nx, nz);
}
