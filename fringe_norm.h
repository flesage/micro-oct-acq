#ifndef FRINGE_NORM_H
#define FRINGE_NORM_H
#include "cuComplex.h"

void fringeNorm(float* fringe, const float* avg_fringe, const float* hann_window, int nx, int nz);
void mag(cuComplex *data, float *mag, int dsz);
void complex_convolve(int nx, int nz, int hpf_npts, cuComplex* c_image, float* kernel, cuComplex* c_filt_image);
void phase_adjacent(int nx, int nz, cuComplex* c_filt_image, float* phase);

#endif // FRINGE_NORM_H

