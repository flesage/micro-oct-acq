/*
 * FringeFFT.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: flesage
 */

#include "FringeFFT.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <nppi.h>
#include "fringe_norm.h"


#define PI ((float)3.1415926)

FringeFFT::FringeFFT() : plan(0), d_fringe(0), d_interpfringe(0), d_signal(0), p_nz(0),
    p_nx (0), handle(0), descr(0), dCsrValA(0), dCsrRowPtrA(0), dCsrColIndA(0), totalNnz(0)
{
    // By default, we use device 0
    int devID = 0;
    cudaError_t error;
    struct cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }
    read_interp_matrix();
    pXmap = 0;
    pYmap = 0;
    cublasCreate_v2(&handle);
    d_hp_filter = 0;

}

FringeFFT::~FringeFFT() {

    cufftDestroy(plan);
    cudaFree(d_signal);
    cudaFree(d_fringe);
    cudaFree(d_interpfringe);
    cudaFree(d_hann_dispcomp);
    cudaFree(d_mean_fringe);
    cudaFree(d_ones);
    cublasDestroy_v2(handle);
    if(d_hp_filter) cudaFree(d_hp_filter);
    // Only delete if we are using sparse module
    if(dCsrValA)
    {
        cudaFree(dCsrValA);
        cudaFree(dCsrRowPtrA);
        cudaFree(dCsrColIndA);
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(handle);
    }
    if(pXmap)
    {
        cudaFree(pXmap);
        cudaFree(pYmap);
        cudaFree(d_nointerp);
        cudaFree(pDst);
    }
}

void FringeFFT::init(int nz, int nx)
{
    p_nz = nz;
    p_nx = nx;

    // Allocate device memory
    cudaMalloc((void **)&d_fringe, p_nz * p_nx * sizeof(cufftReal));
    cudaMalloc((void **)&d_signal, (p_nz/2+1) * p_nx * sizeof(cufftComplex));
    // Allocate space for intermediate interp fringe
    cudaMalloc((void **)&d_interpfringe, p_nz * p_nx * sizeof(cufftReal));
    // Allocate memory for hann/disp comp matrix
    // Default is real hann window
    cudaMalloc((void **)&d_hann_dispcomp, p_nz * sizeof(cuFloatComplex));
    cudaMalloc((void **)&d_mean_fringe, p_nz * sizeof(cuFloatComplex));
    cudaMalloc((void **)&d_ones, p_nx * sizeof(float));

    cuFloatComplex* multiplier = new cuFloatComplex[p_nz];
    for (int i = 0; i < p_nz; i++) {
        multiplier[i].x = 0.5 * (1 - cos(2*PI*i/p_nz));
        multiplier[i].y = 0.0;
    }
    cudaMemcpy(d_hann_dispcomp, (cuFloatComplex*) multiplier, p_nz *sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete [] multiplier;

    // For mean fringe computation
    float* norm=new float[p_nx];
    for (int i = 0; i < p_nx; i++) {
        norm[i]=1.0/p_nx;
    }
    cudaMemcpy(d_ones, (float*) norm, p_nx *sizeof(float), cudaMemcpyHostToDevice);
    delete [] p_nx;

    // CUFFT plan
    cufftPlan1d(&plan, nz, CUFFT_R2C, nx);
}

void set_disp_comp_vect(float* disp_comp_vector)
{
    // Combine hann window with dipersion compensation
    cuFloatComplex* multiplier = new cuFloatComplex[p_nz];
    for (int i = 0; i < p_nz; i++) {
        multiplier[i].x = 0.5 * (1 - cos(2*PI*i/p_nz))*cos(disp_comp_vector[i]);
        multiplier[i].y = 0.5 * (1 - cos(2*PI*i/p_nz))*sin(disp_comp_vector[i]);
    }
    cudaMemcpy(d_hann_dispcomp, (cuFloatComplex*) multiplier, p_nz *sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    delete [] multiplier;
}

void FringeFFT::do_fft(float* in_fringe, cufftComplex* out_signal)
{
    // Copy host memory to device
    cudaMemcpy(d_fringe, (cufftReal*) in_fringe, p_nz * p_nx *sizeof(cufftReal), cudaMemcpyHostToDevice);
    cufftExecR2C(plan, d_fringe, d_signal);
    cudaMemcpy(out_signal, d_signal, (p_nz/2+1) * p_nx * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
}

void FringeFFT::interp_and_do_fft(float* in_fringe, cufftComplex* out_signal)
{
    // Copy host memory to device
    cudaMemcpy(d_fringe, in_fringe, p_nz * p_nx* sizeof(cufftReal), cudaMemcpyHostToDevice);
    // Perform matrix-vector multiplication with the CSR-formatted matrix A
    float alpha = 1.0;
    float beta = 0.0;
    // Do interpolation by sparse matrix multiplication
    cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, p_nz, p_nx, p_nz, totalNnz, &alpha, descr, dCsrValA, dCsrRowPtrA, dCsrColIndA, d_fringe, p_nz, &beta,d_interpfringe,p_nz);
    cudaDeviceSynchronize();
    // Compute reference
    cublasSgemv(CUBLAS_OP_N, p_nz, p_nx, 1.0, d_interpfringe, p_nz, d_ones, 1, 0.0, d_mean_fringe, 1);
    cudaDeviceSynchronize();

    // Multiply by dispersion compensation vector and hann window, store back in d_interpfringe
    fringeNorm(d_interpfringe, d_mean_fringe, d_hann_dispcomp, p_nx, p_nz);
    cudaDeviceSynchronize();

    // Do fft
    cufftExecR2C(plan, d_interpfringe, d_signal);
    cudaDeviceSynchronize();
    cudaMemcpy(out_signal, d_signal, (p_nz/2+1) * p_nx * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
}

void FringeFFT::compute_doppler(float line_period)
{
    //    % FWHM = 2.35482 * sigma
    float sigma=0.2e-3/2.35482;
    PutDopplerHPFilterOnGPU(sigma, lineperiod);

    //NPP call current_frame_hpf = conv2(tmp,filter_kernel,'same');
    //Compute phase angle of adjacent lines following hpf
    //phase = angle(current_frame_hpf(:,1:end-1).*conj(current_frame_hpf(:,2:end)));

    // NPP call Apply convolution window to reduce noise, maybe should be done before taking angle
    //phase=convn(phase,spatial_filter,'same');
    //speed=1313*1e-6*phase/(4*pi*a_line_period);
}

void FringeFFT::PutDopplerHPFilterOnGPU(float sigma, float lineperiod)
{
    //This function will design a filter_kernel for use with the doppler processor function.
    //           - sigma is the width of the gaussian function used in the kernel (in second).
    //           - lineperiod is the time step for each a-line acquisition (in seconds).
    int npts=3*sigma/lineperiod;
    float* filter = new float[2*npts+1];
    float norm = 0.0;
    for(int i=0;i<2*npts+1;i++)
    {
        t=(i-npts)*lineperiod;
        filter[i]=-exp(-t*t/(2*sigma*sigma));
        norm-=filter[i];
    }
    for(int i=0;i<2*npts+1;i++)
    {
        filter[i]=filter[i]/norm;
    }
    filter[0]+=1;
    cudaMalloc((void **)&d_hp_filter, (2*npts+1) * sizeof(float));
    cudaMemcpy(d_hp_filter, filter, (2*npts+1)* sizeof(float), cudaMemcpyHostToDevice);

    delete [] filter;
}

void FringeFFT::read_interp_matrix()
{
    // Read matrix and cast to float as a dense A matrix
    double* p_interpolation_matrix = new double[p_nz*p_nz];
    float* A=new float[p_nz*p_nz];
    FILE* fp=fopen("/Users/flesage/Desktop/interpolation_matrix.dat","rb");
    fread(p_interpolation_matrix,sizeof(double),p_nz*p_nz,fp);
    fclose(fp);
    for (int i=0;i<p_nz*p_nz;i++) A[i]=(float) p_interpolation_matrix[i];
    delete[] p_interpolation_matrix;

    // Transform interpolation matrix into a sparse matrix
    cusparseStatus_t status = cusparseCreate(&handle);

    // Allocate device memory for the matrix A
    int* dNnzPerRow;
    float* dA;
    cudaMalloc((void **)&dA, sizeof(float) * p_nz * p_nz);
    cudaMalloc((void **)&dNnzPerRow, sizeof(int) * p_nz);

    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // Transfer the dense matrix A to the device
    cudaMemcpy(dA, A, sizeof(float) * p_nz * p_nz, cudaMemcpyHostToDevice);
    // Compute the number of non-zero elements in A
    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, p_nz, p_nz, descr, dA, p_nz, dNnzPerRow, &totalNnz);

    // Allocate device memory to store the sparse CSR representation of A
    cudaMalloc((void **)&dCsrValA, sizeof(float) * totalNnz);
    cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (p_nz + 1));
    cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalNnz);

    // Convert A from a dense formatting to a CSR formatting, using the GPU
    cusparseSdense2csr(handle, p_nz, p_nz, descr, dA, p_nz, dNnzPerRow, dCsrValA, dCsrRowPtrA, dCsrColIndA);

    // Cleaning done in destructor but do not need dA anymore
    delete [] A;
    cudaFree(dA);
    cudaFree(dNnzPerRow);

    // After this we can perform multiplications.
}


void FringeFFT::get_radial_img(Npp32f* fringe, Npp32f* interp_fringe)
{
    NppiSize oSrcSize={p_nx,(p_nz/2+1)};
    int nSrcStep = p_nx*sizeof(float);
    NppiRect oSrcROI;
    oSrcROI.height=p_nx;
    oSrcROI.width=(p_nz/2+1);
    NppiSize oDstSizeROI;
    oDstSizeROI.height=(2*p_n_radial_pts+1);
    oDstSizeROI.width=(2*p_n_radial_pts+1);
    // Copy host memory to device
    cudaMemcpy(d_nointerp, fringe, (p_nz/2+1) * p_nx * sizeof(Npp32f), cudaMemcpyHostToDevice);
    //    Remap supports the following interpolation modes:.
    //    NPPI_INTER_NN NPPI_INTER_LINEAR NPPI_INTER_CUBIC NPPI_INTER_CUBIC2P_BSPLINE
    //    NPPI_INTER_CUBIC2P_CATMULLROM NPPI_INTER_CUBIC2P_B05C03 NPPI_INTER_-
    //    LANCZOS
    //    Remap chooses source pixels using pixel coordinates explicitely supplied in two 2D device memory image
    //    arrays pointed to by the pXMap and pYMap pointers. The pXMap array contains the X coordinated and
    //    the pYMap array contains the Y coordinate of the corresponding source image pixel to use as input. These
    //    coordinates are in floating point format so fraction pixel positions can be used. The coordinates of the
    //    source pixel to sample are determined as follows:
    //    nSrcX = pxMap[nDstX, nDstY] nSrcY = pyMap[nDstX, nDstY]
    //    In the Remap functions below source image clip checking is handled as follows:
    //    If the source pixel fractional x and y coordinates are greater than or equal to oSizeROI.x and less than
    //    oSizeROI.x + oSizeROI.width and greater than or equal to oSizeROI.y and less than oSizeROI.y + oSizeROI.height
    //    then the source pixel is considered to be within the source image clip rectangle and the source
    //    image is sampled. Otherwise the source image is not sampled and a destination pixel is not written to the
    //    destination image.
    //Parameters:
    //          pSrc Source-Image Pointer.
    //          oSrcSize Size in pixels of the source image.
    //          nSrcStep Source-Image Line Step.
    //          oSrcROI Region of interest in the source image.
    //        pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling
    //        source image.
    //        nXMapStep pXMap image array line step in bytes.
    //        pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling
    //        source image.
    //        nYMapStep pYMap image array line step in bytes.
    //        pDst Destination-Image Pointer.
    //        nDstStep Destination-Image Line Step.
    //        oDstSizeROI Region of interest size in the destination image.
    //        eInterpolation The type of interpolation to perform resampling
    NppStatus status = nppiRemap_32f_C1R (d_nointerp, oSrcSize, nSrcStep, oSrcROI,
                                          pXmap, (2*p_n_radial_pts+1)*sizeof(Npp32f),
                                          pYmap, (2*p_n_radial_pts+1)*sizeof(Npp32f),
                                          pDst, (2*p_n_radial_pts+1)*sizeof(Npp32f), oDstSizeROI, NPPI_INTER_LINEAR);
    cudaMemcpy(interp_fringe,pDst,(2*p_n_radial_pts+1) * (2*p_n_radial_pts+1) * sizeof(Npp32f), cudaMemcpyDeviceToHost);
}

void FringeFFT::pre_compute_positions(int n_ang_pts, int n_radial_pts)
{
    p_n_radial_pts = n_radial_pts;

    float* theta_vals = new float[(2*n_radial_pts+1)*(2*n_radial_pts+1)];
    float* r_vals = new float[(2*n_radial_pts+1)*(2*n_radial_pts+1)];
    int index=0;
    for(int ix=-n_radial_pts;ix<n_radial_pts+1;ix++)
    {
        for(int iy=-n_radial_pts;iy<n_radial_pts+1;iy++){
            r_vals[index]=sqrt(ix*ix+iy*iy);
            // Get angle between 0-2pi and map to n angular points for future interpolation
            theta_vals[index]=atan(iy/ix);
            if(theta_vals[index]<0) theta_vals[index]+=2*PI;
            theta_vals[index]=theta_vals[index]/(2*PI)*n_ang_pts;
            index = index+1;
        }
    }
    // Allocate gpu position memory and copy values
    cudaMalloc((void **)&pXmap, sizeof(Npp32f)*(2*n_ang_pts+1)*(2*n_ang_pts+1));
    cudaMalloc((void **)&pYmap, sizeof(Npp32f)*(2*n_ang_pts+1)*(2*n_ang_pts+1));
    cudaMemcpy(pXmap, theta_vals, sizeof(Npp32f)*(2*n_ang_pts+1)*(2*n_ang_pts+1), cudaMemcpyHostToDevice);
    cudaMemcpy(pYmap, r_vals, sizeof(Npp32f)*(2*n_ang_pts+1)*(2*n_ang_pts+1), cudaMemcpyHostToDevice);
    delete [] theta_vals;
    delete [] r_vals;

    // Allocate input and dest images
    cudaMalloc((void **)&d_nointerp, (p_nz/2+1) * p_nx * sizeof(Npp32f));
    cudaMalloc((void **)&pDst, sizeof(Npp32f)*(2*n_ang_pts+1)*(2*n_ang_pts+1));


}
