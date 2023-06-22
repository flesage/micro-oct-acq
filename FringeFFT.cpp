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

FringeFFT::FringeFFT(unsigned int n_repeat, int factor) : p_nz(0),
    p_nx (0), p_n_repeat(n_repeat), p_factor(factor), p_fringe(0,0,f32), p_interpfringe(0,0,f32), p_mean_fringe(0,0,f32),p_signal(0,0,f32),
    p_sparse_interp(0,0,f32),  p_hann_dispcomp(0,0,f32), p_phase(0,0,f32),
    p_hp_filter(0,0,f32), p_filt_signal(0,0,c32), p_pos0(0,0,f32), p_pos1(0,0,f32), p_angio_stack(0,0,0,f32), p_angio(0,0,f32),p_struct(0,0,f32), p_norm_signal(0,0,f32)
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
    p_interpfringe = af::array(p_nz,p_nx,c32);
    p_mean_fringe = af::array(p_nz,1,c32);
    p_angio=af::array(p_nz/2,p_nx,f32);
    p_struct=af::array(p_nz/2,p_nx,f32);
    p_angio_stack=af::array(p_nz/2,p_nx,p_n_repeat,f32);
    p_norm_signal=af::constant(0.0,p_nz,p_nx,f32);
    p_angio_algo=0;

    double* tmp=new double[p_nz];
    FILE* fp=fopen("C:\\Users\\Public\\Documents\\filter.dat","rb");
    if(fp == 0)
    {
        std::cerr << "Check if filter file exists" << std::endl;
        exit(-1);
    }
    fread(tmp,sizeof(double),p_nz,fp);

    double* phase=new double[p_nz];
    FILE* fp2=fopen("C:\\Users\\Public\\Documents\\phase.dat","rb");
    if(fp2 == 0)
    {
        std::cerr << "Check if phase file exists" << std::endl;
        exit(-1);
    }
    fread(phase,sizeof(double),p_nz,fp2);
    fclose(fp2);
    float* f_phase = new float[p_nz];
    for(int i=0;i<p_nz;i++) f_phase[i]=(float) phase[i];

    af::cfloat h_unit = {0, 1};  // Host side
    af::array unit_j = af::constant(h_unit, 1, c32);
    af::array h_phase = af::complex(af::array(p_nz,1,f_phase,afHost));
    float* filter = new float[p_nz];
    for(int i=0;i<p_nz;i++) filter[i]=(float) (tmp[i]*0.00001);
    p_hann_dispcomp = af::complex(af::array(p_nz,1,filter,afHost));
    p_hann_dispcomp *= af::exp(tile(unit_j(0), h_phase.dims())*h_phase);

    delete [] f_phase;
    delete [] phase;
    delete [] filter;
    delete [] tmp;
}

void FringeFFT::set_disp_comp_vect(float* disp_comp_vector)
{
    // Should be more careful since 2 calls could modify this wrongly but
    // the ImageViewer is restarted on each acquisition, so this is only called once.
    // Correct when cleaning.
    af::cfloat h_unit;  // Host side
    h_unit.real = 0;
    h_unit.imag = 1;
    af::array unit_j = af::constant(h_unit, 1, c32);
    af::array phase = af::complex(af::array(p_nz,1,disp_comp_vector,afHost));
    std::cerr << "Here: " << p_nz << std::endl;

    p_hann_dispcomp *= af::exp(-unit_j*phase);
    std::cerr << "Vecteur de compensation fait pour calcul sur gpu" << std::endl;
}

void FringeFFT::interp_and_do_fft(unsigned short* in_fringe,unsigned char* out_signal, float p_image_threshold, float p_hanning_threshold)
{
    // Interpolation by sparse matrix multiplication
    af::dim4 dims(2048,p_nx,1,1);
    af::array tmp(p_nz,p_nx,in_fringe,afHost);
    p_interpfringe = matmul(p_sparse_interp.as(c32),tmp.as(c32));

    // Compute reference
    p_mean_fringe = mean(p_interpfringe, 1);

    // Multiply by dispersion compensation vector and hann window, store back in p_interpfringe
    gfor (af::seq i, p_nx)
            //p_interpfringe(af::span,i)=((p_interpfringe(af::span,i)-p_mean_fringe(af::span)/(p_mean_fringe(af::span)+p_hanning_threshold)))*p_hann_dispcomp;
            p_interpfringe(af::span,i)=(p_interpfringe(af::span,i)-p_mean_fringe(af::span))*p_hann_dispcomp;

    // Do fft
    p_signal = af::fft(p_interpfringe);
    // Since it is now a complex fft, only keep half the values (positive freqs)
    p_signal = p_signal.rows(1,1024);
    // Here we have the complex signal available, compute its magnitude, take log on GPU to go faster
    // Transfer half as much data back to CPU
    p_norm_signal = af::log(af::abs(p_signal)+p_image_threshold);
    float l_max = af::max<float>(p_norm_signal);
    float l_min = af::min<float>(p_norm_signal);
    p_norm_signal=255.0*(p_norm_signal-l_min)/(l_max-l_min);
    p_norm_signal.as(u8).host(out_signal);
}

void FringeFFT::image_reconstruction(unsigned short* in_fringe, float* out_data, int p_top_z, int p_bottom_z, float p_hanning_threshold )
{
    // Interpolation by sparse matrix multiplication
    af::dim4 dims(2048, p_nx, 1, 1);
    af::array tmp(p_nz, p_nx, in_fringe, afHost);
    p_interpfringe = matmul(p_sparse_interp.as(c32), tmp.as(c32));

    // Compute reference
    p_mean_fringe = mean(p_interpfringe, 1);

    // Multiply by dispersion compensation vector and hann window, store back in p_interpfringe
    gfor (af::seq i, p_nx)
        //p_interpfringe(af::span,i)=((p_interpfringe(af::span,i)-p_mean_fringe(af::span)/(p_mean_fringe(af::span)+p_hanning_threshold)))*p_hann_dispcomp;
        p_interpfringe(af::span,i)=(p_interpfringe(af::span,i)-p_mean_fringe(af::span))*p_hann_dispcomp;

    // Do fft
    p_signal = af::abs(af::fft(p_interpfringe));

    // Crop the bscan
    p_signal = p_signal.rows(p_top_z, p_bottom_z);

    // Set as output
    p_signal.host(out_data);
}

void FringeFFT::setAngioAlgo(int angio_algo)
{
    p_angio_algo=angio_algo;
}

void FringeFFT::get_angio(unsigned short* in_fringe,af::array* out_data, float p_image_threshold, float p_hanning_threshold)
{
    // Interpolation by sparse matrix multiplication
    af::dim4 dims(2048,p_nx,1,1);
    af::array tmp(p_nz,p_nx*p_n_repeat,in_fringe,afHost);
    p_interpfringe = matmul(p_sparse_interp,tmp.as(f32));
    // Compute reference
    p_mean_fringe = mean(p_interpfringe,1);

    // Multiply by dispersion compensation vector and hann window, store back in d_interpfringe
    gfor (af::seq i, p_nx*p_n_repeat)
            p_interpfringe(af::span,i)=((p_interpfringe(af::span,i)-p_mean_fringe(af::span))/(p_mean_fringe(af::span)+p_hanning_threshold))*p_hann_dispcomp;

    // Do fft
    p_signal = af::fftR2C<1>(p_interpfringe, dims);
    p_signal = af::abs(p_signal.rows(1,af::end));
    p_angio_stack=af::moddims(p_signal,p_nz/2,p_nx,p_n_repeat);

    switch(p_angio_algo)
    {
    case 0:
    {
        for(unsigned int i=0 ; i<(p_n_repeat); i++)
        {
            if(i>0)
            {
                if(i==1)
                {
                    p_angio=af::abs(af::abs(p_angio_stack(af::span,af::span,i))-af::abs(p_angio_stack(af::span,af::span,i-1)));
                    p_angio=p_angio/(p_n_repeat-1);

                } else
                {
                    p_angio=p_angio+(af::abs(af::abs(p_angio_stack(af::span,af::span,i))-af::abs(p_angio_stack(af::span,af::span,i-1))))/(p_n_repeat-1);
                }
            }
        }
        p_norm_signal=meanShift(p_angio, 1, 1, 1);
        break;
    }
    case 1:
    {
        for(unsigned int i=0 ; i<(p_n_repeat); i++)
        {
            if(i>0)
            {
                if(i==1)
                {
                    p_angio=af::abs((p_angio_stack(af::span,af::span,i))-(p_angio_stack(af::span,af::span,i-1)));
                    p_angio=p_angio/(p_n_repeat-1);

                } else
                {
                    p_angio=p_angio+(af::abs(af::abs(p_angio_stack(af::span,af::span,i))-af::abs(p_angio_stack(af::span,af::span,i-1))))/(p_n_repeat-1);
                }
            }
        }
        //p_norm_signal=(p_angio);//(p_struct+(p_image_threshold*100));
        p_norm_signal=meanShift(p_angio, 1, 1, 1);
        break;
    }
    case 2:
    {
        p_norm_signal=af::log(af::var(p_angio_stack,0,2)+p_image_threshold);
        p_norm_signal=meanShift(p_norm_signal, 1, 1, 1);
        break;
    }
    case 3:
    {
        p_norm_signal=20*af::log(af::mean(p_angio_stack,2)+p_image_threshold);
        p_norm_signal=meanShift(p_norm_signal, 1, 1, 1);

        break;
    }
    }
    *out_data = p_norm_signal;
    return;
}

void FringeFFT::compute_hilbert(unsigned short* in_fringe,unsigned char* out_data, float p_hanning_threshold)
{
    // Interpolation by sparse matrix multiplication
    af::dim4 dims(2048,p_nx,1,1);
    af::array tmp(p_nz,p_nx,in_fringe,afHost);
    p_interpfringe = matmul(p_sparse_interp,tmp.as(f32));
    // Compute reference
    p_mean_fringe = mean(p_interpfringe,1);

    // Multiply by dispersion compensation vector and hann window, store back in d_interpfringe
    gfor (af::seq i, p_nx)
            p_interpfringe(af::span,i)=((p_interpfringe(af::span,i)-p_mean_fringe(af::span))/(p_mean_fringe(af::span)+p_hanning_threshold))*p_hann_dispcomp;

    // Do hilbert
    p_signal = af::fft(p_interpfringe);
    p_signal(af::seq(1025,2047),af::span,0,0)=0;
    p_signal(af::seq(1,1023),af::span,0,0)=2*p_signal(af::seq(1,1023),af::span,0,0);
    p_signal=ifft(p_signal);
    af::array wrapped_angle=af::atan2(af::imag(p_signal),af::real(p_signal));

    // On cree les dims dans l'espace k
    af::array n = af::range(dims,0,f32);
    n(0,af::span) = 1/sqrt(2048);
    p_coord = n*n;

    af::array angle = unwrap(wrapped_angle);
    // Here we have the phase signal available, compute its magnitude, take log on GPU to go faster
    // Transfer half as much data back to CPU
    float l_max = af::max<float>(angle);
    float l_min = af::min<float>(angle);
    angle=255.0*(angle-l_min)/(l_max-l_min);
    angle.as(u8).host(out_data);
}

void FringeFFT::init_doppler(float msec_fwhm, float line_period, float spatial_fwhm_um)
{
    // FWHM = 2.35482 * sigma
    float sigma= (float) (msec_fwhm/2.35482);
    PutDopplerHPFilterOnGPU(sigma, line_period);
    p_line_period=line_period;
    p_phase=af::array(p_nz/2+1,p_nx-1,f32);
    p_spatial_fwhm_um = spatial_fwhm_um;
}

void FringeFFT::compute_doppler( unsigned short* in_fringe, unsigned char* out_doppler, float p_image_threshold, float p_hanning_threshold)
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
            p_interpfringe(af::span,i)=((p_interpfringe(af::span,i)-p_mean_fringe(af::span))/(p_mean_fringe(af::span)+p_hanning_threshold))*p_hann_dispcomp;

    // Do fft
    p_signal = af::fftR2C<1>(p_interpfringe, dims);

    // We assume we already have a complex image on the GPU (p_signal) and start from there.
    // thus this needs to always be called after interp_an_do_fft.
    p_filt_signal=convolve(p_signal,p_hp_filter);
    //p_filt_signal = p_signal;
    float speed_factor=float(1313*1e-6/(4*PI*p_line_period*1.33));
    p_phase=speed_factor*arg(p_filt_signal.cols(1,af::end)*conjg(p_filt_signal.cols(0,af::end-1)));
    p_phase = convolve(p_phase,af::gaussianKernel(n_gauss_z,n_gauss_x));
    float l_max = af::max<float>(p_phase);
    float l_min = af::min<float>(p_phase);
    p_phase=255*(p_phase-l_min)/(l_max-l_min);
    p_phase = p_phase.rows(1,af::end);
    p_phase.as(u8).host(out_doppler);
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

af::array FringeFFT::unwrap(const af::array& angle)
{
    return angle + af::round( ( laplacian( af::cos(angle)*laplacian(af::sin(angle),false) -
                                           af::sin(angle)*laplacian(af::cos(angle),false), true ) - angle ) / 2/af::Pi ) * 2*af::Pi;
}

// Laplace operator and inverse
af::array FringeFFT::laplacian(const af::array& arr, bool inverse)
{
    if(inverse)
        return idct1(dct1(arr)/p_coord); // factor -M*N/4/pi**2 omitted
    else
        return idct1(dct1(arr)*p_coord); // factor -4*pi**2/M/N omitted
}

af::array FringeFFT::dct1(const af::array& arr)
{
    af::cfloat h_unit;
    h_unit.real = 0.0;
    h_unit.imag = 1.0;
    //af::array unit = af::constant(h_unit, 1, c32);

    int N = (int) arr.dims(0);
    af::array out=arr.copy();
    out = 2 * af::real(af::exp(-0.5*h_unit*af::Pi/N*af::range(arr.dims(),0,arr.type())) *
                       af::fft(af::join(0,arr(af::seq(0,N-1,2),af::span),af::flip(arr(af::seq(1,N-1,2),af::span),0))) );
    out /= sqrt(2*N);
    out(0,af::span) /= sqrt(2);
    return out;
}


af::array FringeFFT::idct1(const af::array& arr)
{
//    af::cfloat h_unit;
//    h_unit.real = 0.0;
//    h_unit.imag = 1.0;
//    //af::array unit = af::constant(h_unit, 1, c32);
//    int N = (int) arr.dims(0);
//    af::array tmp = arr.copy();
//    af::array offset = af::tile(tmp(0,af::span),N);
//    tmp(0,af::span) = 0.;
//    tmp = 2 * af::real( af::ifft( af::exp(0.5*h_unit*af::Pi/N*af::range(arr.dims(),0,arr.type())) * tmp ) * N );
//
//    out(af::seq(0,N-1,2),af::span) = tmp(af::seq(0,floor(N/2)-1),af::span);
//    out(af::seq(1,N-1,2),af::span) = af::flip(tmp(af::seq(floor(N/2),af::end),af::span),0);
//    offset /= sqrt(N);
//    out /= sqrt(2*N);
//    out += offset;
    af::array out = af::constant(0,arr.dims(),arr.type());
    //tmp = offset = None;
    return out;
}
