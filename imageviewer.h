#ifndef ImageViewer_H
#define ImageViewer_H

#include <QImage>
#include <QLabel>
#include <QKeyEvent>
#include "FringeFFT.h"
#include "fwhmviewer.h"
#include "angioviewer3dform.h"
#include "arrayfire.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include "asdkDM.h"

using namespace acs;

class QScrollBar;



class ImageViewer : public QLabel
{
    enum ViewMode
    {
        FRINGE = 0,
        STRUCT = 1,
        DOPPLER = 2,
        HILBERT = 3,
        ANGIO = 4
    };
    Q_OBJECT
public:
    explicit ImageViewer(QWidget *parent = 0, int n_alines = 100, int n_extra = 40, int ny=1, int view_depth=512, unsigned int n_repeat=1, float msec_fwhm=0.0002, float spatial_fwhm=3.5,
                         float line_period=0.01, float dimz=3.5, float dimx=3.5, int factor=1, DM* in_dm=0, float** in_zern=0, int in_z_min=2, int in_z_max=13 );
    virtual ~ImageViewer();
    virtual int heightForWidth( int width ) const;
    virtual QSize sizeHint() const;
    void put(unsigned short* frame);
    void set_disp_comp_vect(float* disp_comp_vector);
signals:
    void sig_updateImageThreshold(float);
    void sig_updateLineScanPos(int,int,int,int);
    void sig_updateAverageAngio(bool);

public slots:
    void updateView();
    void resizeEvent(QResizeEvent *);
    void setPixmap ( const QPixmap & );
    void updateImageThreshold(float);
    void updateHanningThreshold(float);
    void updateAngioAlgo(int);
    void setCurrentViewModeStruct();
    void checkLine(bool,int,int);
    void updateLineScanPos(int start_x, int start_y, int stop_x, int stop_y);
    void updateViewLinePositions(bool, int, int);
    void updateAngioAverageFlag(bool);
    void turnDMOn(void);
    void turnDMOff(void);
    void optimizeDM(void);
    void resetDM(void);
    void setMetric(int);
    void setPercent(double);

protected:
    virtual void  keyPressEvent(QKeyEvent *event);

private:

    double scaleFactor;
    int p_n_alines;
    int p_ny;
    int p_factor;
    int p_n_repeat;
    int p_n_extra;
    FringeFFT f_fft;
    QImage p_image;
    QImage p_fringe_image;
    QImage p_hilbert_image;
    QImage p_doppler_image;
    float p_image_threshold;
    float p_hanning_threshold;
    int p_angio_algo;
    unsigned short int* p_data_buffer;
    QMutex p_mutex;
    bool is_optimization;
    bool is_focus_line;
    QPixmap pix;
    double* real_fringe;
    ViewMode p_current_viewmode;
    FWHMViewer* p_fwhm_view;
    FWHMViewer* p_phase_view;
    AngioViewer3DForm* p_angio_view;
    int p_view_depth;
    bool p_line_status;
    int p_start_line;
    int p_stop_line;
    unsigned int p_frame_number;
    af::array p_angio;
    bool p_angio_averageFlag;

    DM *dm;
    float** Z2C;
    int z_idx;
    int z_min;
    int z_max;
    Scalar* dm_data;
    Scalar* dm_current_opt;
    double dm_metric[50];
    double dm_metric_max;
    int dm_idx;
    int dm_idx_max;
    double dm_c[50];
    bool is_dm_optimization;
    int p_metric;
    double p_percent;
    double getMetric(QImage image, int metric_number);
    void moveDM(Scalar* data, double amp);
    double polyfit();
    void optimizeDM(QImage image);
};

#endif // ImageViewer_H
