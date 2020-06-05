#ifndef ImageViewer_H
#define ImageViewer_H

#include <QImage>
#include <QLabel>
#include <QKeyEvent>
#include "FringeFFT.h"
#include "fwhmviewer.h"
#include "angioviewer3dform.h"
#include "arrayfire.h"

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
    explicit ImageViewer(QWidget *parent = 0, int n_alines = 100, int ny=1, int view_depth=512, unsigned int n_repeat=1, float msec_fwhm=0.0002, float spatial_fwhm=3.5,
                         float line_period=0.01, float dimz=3.5, float dimx=3.5, int factor=1);
    virtual ~ImageViewer();
    virtual int heightForWidth( int width ) const;
    virtual QSize sizeHint() const;
    void put(unsigned short* frame);
signals:
    void sig_updateImageThreshold(float);
    void sig_updateLineScanPos(int,int,int,int);
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

protected:
    virtual void  keyPressEvent(QKeyEvent *event);

private:
    FringeFFT f_fft;
    double scaleFactor;
    int p_n_alines;
    int p_factor;
    int p_n_repeat;
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
    int p_ny;
    af::array p_angio;
};

#endif // ImageViewer_H
