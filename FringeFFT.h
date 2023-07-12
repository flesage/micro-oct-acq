/*
 * FringeFFT.h
 *
 *  Created on: Dec 12, 2016
 *      Author: flesage
 */

#ifndef FRINGEFFT_H_
#define FRINGEFFT_H_


#include "arrayfire.h"

class FringeFFT {
public:
    FringeFFT(unsigned int n_repeat, int factor);
	virtual ~FringeFFT();
    void init( int nz, int nx, float dimz, float dimx);
    void set_disp_comp_vect(float* disp_comp_vector);
    void interp_and_do_fft(unsigned short* in_fringe,unsigned char* out_data, float p_image_threshold, float p_hanning_threshold);
    void image_reconstruction(unsigned short* in_fringe, float* out_data, int p_top_z, int p_bottom_z, float p_hanning_threshold );
    void get_angio(unsigned short* in_fringe,af::array* out_data, float p_image_threshold, float p_hanning_threshold);
    void compute_hilbert(unsigned short* in_fringe,unsigned char* out_data, float p_hanning_threshold);
    void init_doppler(float msec_fwhm, float line_period, float spatial_fwhm_um);
    void PutDopplerHPFilterOnGPU(float sigma, float lineperiod);
    void compute_doppler(unsigned short* in_fringe, unsigned char *out_doppler, float p_image_threshold, float p_hanning_threshold);
	void read_interp_matrix();
    void pre_compute_positions(int n_ang_pts, int n_radial_pts);
    void get_radial_img(unsigned short* in_fringe, float* out_image);
    void setAngioAlgo(int angio_algo);
private:
    af::array unwrap(const af::array& angle);
    af::array laplacian(const af::array& arr, bool inverse);
    af::array dct1(const af::array& arr);
    af::array idct1(const af::array& arr);
    int p_nz;
    int p_nx;
    int p_angio_algo;
    unsigned int p_n_repeat;
    int p_factor;
    float p_dimx;
    float p_dimz;
    af::array p_fringe;
    af::array p_interpfringe;
    af::array p_signal;
    af::array p_filt_signal;
    af::array p_phase;
    af::array p_hann_dispcomp;
    af::array p_mean_fringe;
    af::array p_hp_filter;
    af::array p_sparse_interp;
    af::array p_pos0;
    af::array p_pos1;
    af::array p_angio_stack;
    af::array p_angio;
    af::array p_struct;
    af::array p_norm_signal;
    af::array p_background;
    int p_hpf_npts;
    float p_line_period;
    int p_n_radial_pts;
    float p_spatial_fwhm_um;
    af::array p_coord;
};

#endif /* FRINGEFFT_H_ */
