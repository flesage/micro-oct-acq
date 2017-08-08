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
    p_hp_filter(0,0,f32), p_filt_signal(0,0,c32), p_pos0(0,0,f32), p_pos1(0,0,f32)
{
    p_hpf_npts=0;
}

FringeFFT::~FringeFFT() {
}

void FringeFFT::init(int nz, int nx, float dimz, float dimx)
{
    p_nz = nz;
    p_nx = nx;
    p_dimz=dimz;
    p_dimx=dimx;
    read_interp_matrix();

    p_fringe = af::array(p_nz,p_nx,f32);
    p_interpfringe = af::array(p_nz,p_nx,f32);
    p_mean_fringe = af::array(p_nz,1,f32);

    double* tmp=new double[p_nz];
    FILE* fp=fopen("C:\\Users\\Public\\Documents\\filter.dat","rb");

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

void FringeFFT::interp_and_do_fft(unsigned short* in_fringe, unsigned char* out_signal, float p_threshold)
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
    af::array p_norm_signal = log(abs(p_signal)+p_threshold);
    float l_max = af::max<float>(p_norm_signal);
    float l_min = af::min<float>(p_norm_signal);
    p_norm_signal=255*(p_norm_signal-l_min)/(l_max-l_min);
    p_norm_signal.as(u8).T().host(out_signal);
}

void FringeFFT::init_doppler(float msec_fwhm, float line_period, float spatial_fwhm_um)
{
    // FWHM = 2.35482 * sigma
    float sigma= (float) (fwhm/2.35482);
    PutDopplerHPFilterOnGPU(sigma, line_period);
    p_line_period=line_period;
    p_phase=af::array(p_nz/2+1,p_nx-1,f32);
    p_spatial_fwhm_um = spatial_fwhm_um;
}
void FringeFFT::compute_doppler( unsigned short* in_fringe, unsigned char* out_doppler)
{
    int n_gauss_x = (int) (p_spatial_fwhm_um/p_dimx);
    if(n_gauss_x == 0) n_gauss_x=1;
    int n_gauss_z = (int) (p_spatial_fwhm_um/p_dimz);
    if(n_gauss_z == 0) n_gauss_z=1;

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

    // We assume we already have a complex image on the GPU (p_signal) and start from there.
	// thus this needs to always be called after interp_an_do_fft.
    //p_filt_signal=convolve(p_signal,p_hp_filter);
    p_filt_signal = p_signal;
    float speed_factor=1313*1e-6/(4*PI*p_line_period*1.33);
    p_phase=speed_factor*arg(p_filt_signal.cols(1,af::end)*conjg(p_filt_signal.cols(0,af::end-1)));
    p_phase = convolve(p_phase,af::gaussianKernel(n_gauss_z,n_gauss_x));
    float l_max = af::max<float>(p_phase);
    float l_min = af::min<float>(p_phase);
    p_phase=255*(p_phase-l_min)/(l_max-l_min);
    p_phase.as(u8).T().host(out_doppler);
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

    p_hp_filter = af::array(1,p_hpf_npts,filter,afHost);

    delete [] filter;
}

void FringeFFT::read_interp_matrix()
{
    // Read matrix and cast to float as a dense A matrix
    double* p_interpolation_matrix = new double[p_nz*p_nz];
    float* A=new float[p_nz*p_nz];
    FILE* fp=fopen("C:\\Users\\Public\\Documents\\interpolation_matrix.dat","rb");

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


void FringeFFT::get_radial_img(unsigned short* in_fringe, float* out_image)
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

    af::approx2(log(abs(p_signal)),p_pos0,p_pos1).host(out_image);
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
            r_vals[index]=(float) sqrt(1.0*ix*ix+1.0*iy*iy);
            // Get angle between 0-2pi and map to n angular points for future interpolation
            if(ix != 0)
            {
                theta_vals[index]=(float) atan(1.0*iy/ix);
            }
            else
            {
                if(iy > 0)
                    theta_vals[index]=PI/2;
                else
                    theta_vals[index]=-PI/2;
            }

            if(theta_vals[index]<0) theta_vals[index]+=2*PI;
            theta_vals[index]=theta_vals[index]/(2*PI)*(n_ang_pts-1);
            index = index+1;
        }
    }
    p_pos0 = af::array((2*n_radial_pts+1),(2*n_radial_pts+1), r_vals, afHost);
    p_pos1 = af::array((2*n_radial_pts+1),(2*n_radial_pts+1), theta_vals, afHost);

    delete [] theta_vals;
    delete [] r_vals;

}
