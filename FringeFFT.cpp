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


#define PI ((float)3.1415926)

FringeFFT::FringeFFT() : p_nz(0),
    p_nx (0), p_interpfringe(0,0,f32), p_fringe(0,0,f32), p_sparse_interp(0,0,f32),
    p_mean_fringe(0,0,f32), p_signal(0,0,f32), p_hann_dispcomp(0,0,f32), p_phase(0,0,f32),
    p_hp_filter(0,0,f32), p_filt_signal(0,0,c32)
{
    p_hpf_npts=0;
}

FringeFFT::~FringeFFT() {
}

void FringeFFT::init(int nz, int nx)
{
    p_nz = nz;
    p_nx = nx;
    read_interp_matrix();

    p_fringe = af::array(p_nz,p_nx,f32);
    p_interpfringe = af::array(p_nz,p_nx,f32);
    p_mean_fringe = af::array(p_nz,1,f32);

    double* tmp=new double[p_nz];
    FILE* fp=fopen("C:\\Users\\Public\\Documents\\filter.dat","rb");
    //FILE* fp=fopen("/Users/flesage/Documents/data/oct_poly_datfiles/filter.dat","rb");

    if(fp == 0)
    {
        std::cerr << "Check if filter file exists" << std::endl;
        exit(-1);
    }
    fread(tmp,sizeof(double),p_nz,fp);
    fclose(fp);
    float* filter = new float[p_nz];
    for(int i=0;i<p_nz;i++) filter[i]=(float) tmp[i];
    p_hann_dispcomp = af::array(p_nz,1,filter,afHost);
    delete [] filter;
    delete [] tmp;
}

void FringeFFT::set_disp_comp_vect(float* disp_comp_vector)
{

}

void FringeFFT::interp_and_do_fft(unsigned short* in_fringe, float* out_signal)
{

    // Interpolation by sparse matrix multiplication
    af::dim4 dims(2048,p_nx,1,1);
    af::array tmp(p_nz,p_nx,in_fringe,afHost);
    p_interpfringe = matmul(p_sparse_interp,tmp.as(f32));
    // Compute reference
    p_mean_fringe = mean(p_interpfringe,1);

    // Multiply by dispersion compensation vector and hann window, store back in d_interpfringe
    gfor (af::seq i, p_nx)
            p_interpfringe(af::span,i)=((p_interpfringe(af::span,i)-p_mean_fringe(af::span))/(p_mean_fringe(af::span)+1e-6))*p_hann_dispcomp;

    // Do fft
    p_signal = af::fftR2C<1>(p_interpfringe, dims);

    // Here we have the complex signal available, compute its magnitude, take log on GPU to go faster
    // Transfer half as much data back to CPU
    log(abs(p_signal)).host(out_signal);
}

void FringeFFT::init_doppler(float fwhm, float line_period)
{
    // FWHM = 2.35482 * sigma
    float sigma= (float) (fwhm/2.35482);
    PutDopplerHPFilterOnGPU(sigma, line_period);
    p_phase=af::array(p_nz/2+1,p_nx-1,f32);
}
void FringeFFT::compute_doppler(float* doppler_signal)
{
    // We assume we already have a complex image on the GPU (p_signal) and start from there.
	// thus this needs to always be called after interp_an_do_fft.
    p_filt_signal=convolve(p_signal,p_hp_filter);
    //float speed_factor=1313*1e-6/(4*PI*a_line_period);
    p_phase=arg(p_filt_signal.cols(1,af::end)*conjg(p_filt_signal.cols(0,af::end-1)));

    p_phase.host(doppler_signal);
    convolve(p_phase,af::gaussianKernel(3,3)).host(doppler_signal);
}

void FringeFFT::PutDopplerHPFilterOnGPU(float sigma, float lineperiod)
{
    //	This function will design a filter_kernel for use with the Doppler processor function.
    //           - sigma is the width of the gaussian function used in the kernel (in second).
    //           - lineperiod is the time step for each a-line acquisition (in seconds).
    int npts=(int) (3*sigma/lineperiod);
    p_hpf_npts=2*npts+1;
    float* filter = new float[p_hpf_npts];
    float norm = 0.0;
    for(int i=0;i<p_hpf_npts;i++)
    {
        float t=(i-npts)*lineperiod;
        filter[i]=-exp(-t*t/(2*sigma*sigma));
        norm-=filter[i];
    }
    for(int i=0;i<p_hpf_npts;i++)
    {
        filter[i]=filter[i]/norm;
    }
    filter[0] += 1;

    p_hp_filter = af::array(p_hpf_npts,1,filter,afHost);

    delete [] filter;
}

void FringeFFT::read_interp_matrix()
{
    // Read matrix and cast to float as a dense A matrix
    double* p_interpolation_matrix = new double[p_nz*p_nz];
    float* A=new float[p_nz*p_nz];
    FILE* fp=fopen("C:\\Users\\Public\\Documents\\interpolation_matrix.dat","rb");
    //FILE* fp=fopen("/Users/flesage/Documents/data/oct_poly_datfiles/interpolation_matrix.dat","rb");

    if(fp == 0)
    {
    	std::cerr << "Check if interpolation file exists" << std::endl;
    	exit(-1);
    }
    fread(p_interpolation_matrix,sizeof(double),p_nz*p_nz,fp);
    fclose(fp);
    for (int i=0;i<p_nz*p_nz;i++) A[i]=(float) p_interpolation_matrix[i];
    delete[] p_interpolation_matrix;
    af::array tmp(p_nz, p_nz, A, afHost);
    p_sparse_interp = sparse(tmp);
}


//void FringeFFT::get_radial_img(Npp32f* fringe, Npp32f* interp_fringe)
//{
//    NppiSize oSrcSize={p_nx,(p_nz/2+1)};
//    int nSrcStep = p_nx*sizeof(float);
//    NppiRect oSrcROI;
//    oSrcROI.height=(p_nz/2+1);
//    oSrcROI.width=p_nx;
//    NppiSize oDstSizeROI;
//    oDstSizeROI.height=(2*p_n_radial_pts+1);
//    oDstSizeROI.width=(2*p_n_radial_pts+1);
//    // Copy host memory to device
//    cudaMemcpy(d_nointerp, fringe, (p_nz/2+1) * p_nx * sizeof(Npp32f), cudaMemcpyHostToDevice);
//    //    Remap supports the following interpolation modes:.
//    //    NPPI_INTER_NN NPPI_INTER_LINEAR NPPI_INTER_CUBIC NPPI_INTER_CUBIC2P_BSPLINE
//    //    NPPI_INTER_CUBIC2P_CATMULLROM NPPI_INTER_CUBIC2P_B05C03 NPPI_INTER_-
//    //    LANCZOS
//    //    Remap chooses source pixels using pixel coordinates explicitely supplied in two 2D device memory image
//    //    arrays pointed to by the pXMap and pYMap pointers. The pXMap array contains the X coordinated and
//    //    the pYMap array contains the Y coordinate of the corresponding source image pixel to use as input. These
//    //    coordinates are in floating point format so fraction pixel positions can be used. The coordinates of the
//    //    source pixel to sample are determined as follows:
//    //    nSrcX = pxMap[nDstX, nDstY] nSrcY = pyMap[nDstX, nDstY]
//    //    In the Remap functions below source image clip checking is handled as follows:
//    //    If the source pixel fractional x and y coordinates are greater than or equal to oSizeROI.x and less than
//    //    oSizeROI.x + oSizeROI.width and greater than or equal to oSizeROI.y and less than oSizeROI.y + oSizeROI.height
//    //    then the source pixel is considered to be within the source image clip rectangle and the source
//    //    image is sampled. Otherwise the source image is not sampled and a destination pixel is not written to the
//    //    destination image.
//    //Parameters:
//    //          pSrc Source-Image Pointer.
//    //          oSrcSize Size in pixels of the source image.
//    //          nSrcStep Source-Image Line Step.
//    //          oSrcROI Region of interest in the source image.
//    //        pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling
//    //        source image.
//    //        nXMapStep pXMap image array line step in bytes.
//    //        pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling
//    //        source image.
//    //        nYMapStep pYMap image array line step in bytes.
//    //        pDst Destination-Image Pointer.
//    //        nDstStep Destination-Image Line Step.
//    //        oDstSizeROI Region of interest size in the destination image.
//    //        eInterpolation The type of interpolation to perform resampling
//    NppStatus status = nppiRemap_32f_C1R (d_nointerp, oSrcSize, nSrcStep, oSrcROI,
//                                          pXmap, (2*p_n_radial_pts+1)*sizeof(Npp32f),
//                                          pYmap, (2*p_n_radial_pts+1)*sizeof(Npp32f),
//                                          pDst, (2*p_n_radial_pts+1)*sizeof(Npp32f), oDstSizeROI, NPPI_INTER_LINEAR);
//    cudaMemcpy(interp_fringe,pDst,(2*p_n_radial_pts+1) * (2*p_n_radial_pts+1) * sizeof(Npp32f), cudaMemcpyDeviceToHost);
//}

//void FringeFFT::pre_compute_positions(int n_ang_pts, int n_radial_pts)
//{
//    p_n_radial_pts = n_radial_pts;
//    float* theta_vals = new float[(2*n_radial_pts+1)*(2*n_radial_pts+1)];
//    float* r_vals = new float[(2*n_radial_pts+1)*(2*n_radial_pts+1)];
//    int index=0;
//    for(int ix=-n_radial_pts;ix<n_radial_pts+1;ix++)
//    {
//        for(int iy=-n_radial_pts;iy<n_radial_pts+1;iy++){
//            r_vals[index]=(float) sqrt(ix*ix+iy*iy);
//            // Get angle between 0-2pi and map to n angular points for future interpolation
//            if(ix != 0)
//            {
//                theta_vals[index]=(float) atan(iy/ix);
//            }
//            else
//            {
//                if(iy > 0)
//                    theta_vals[index]=PI/2;
//                else
//                    theta_vals[index]=-PI/2;
//            }

//            if(theta_vals[index]<0) theta_vals[index]+=2*PI;
//            theta_vals[index]=theta_vals[index]/(2*PI)*n_ang_pts;
//            index = index+1;
//        }
//    }

//    // Allocate gpu position memory and copy values
//    cudaMalloc((void **)&pXmap, sizeof(Npp32f)*(2*n_ang_pts+1)*(2*n_ang_pts+1));
//    cudaMalloc((void **)&pYmap, sizeof(Npp32f)*(2*n_ang_pts+1)*(2*n_ang_pts+1));
//    cudaMemcpy(pXmap, theta_vals, sizeof(Npp32f)*(2*n_ang_pts+1)*(2*n_ang_pts+1), cudaMemcpyHostToDevice);
//    cudaMemcpy(pYmap, r_vals, sizeof(Npp32f)*(2*n_ang_pts+1)*(2*n_ang_pts+1), cudaMemcpyHostToDevice);
//    delete [] theta_vals;
//    delete [] r_vals;

//    // Allocate input and dest images
//    cudaMalloc((void **)&d_nointerp, (p_nz/2+1) * p_nx * sizeof(Npp32f));
//    cudaMalloc((void **)&pDst, sizeof(Npp32f)*(2*n_ang_pts+1)*(2*n_ang_pts+1));


//}
