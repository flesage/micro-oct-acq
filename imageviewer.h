#ifndef ImageViewer_H
#define ImageViewer_H

#include <QImage>
#include <QLabel>
#include <QKeyEvent>
#include "FringeFFT.h"
#include "fwhmviewer.h"

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
    explicit ImageViewer(QWidget *parent = 0, int n_alines = 100, int view_depth=512, unsigned int n_repeat=1, float msec_fwhm=0.0002, float spatial_fwhm=3.5,
                         float line_period=0.01, float dimz=3.5, float dimx=3.5);
    virtual ~ImageViewer();
    virtual int heightForWidth( int width ) const;
    virtual QSize sizeHint() const;
    void put(unsigned short* frame);
signals:

public slots:
    void updateView();
    void resizeEvent(QResizeEvent *);
    void setPixmap ( const QPixmap & );
    void updateImageThreshold(float);
    void updateHanningThreshold(float);

protected:
    virtual void  keyPressEvent(QKeyEvent *event);

private:
    FringeFFT f_fft;
    double scaleFactor;
    int p_n_alines;
    QImage p_image;
    QImage p_fringe_image;
    QImage p_hilbert_image;
    QImage p_doppler_image;
    float p_image_threshold;
    float p_hanning_threshold;
    unsigned short int* p_data_buffer;
    QMutex p_mutex;
    bool is_optimization;
    bool is_focus_line;
    QPixmap pix;
    double* real_fringe;
    ViewMode p_current_viewmode;
    FWHMViewer* p_fwhm_view;
    FWHMViewer* p_phase_view;
    int p_view_depth;
};

#endif // ImageViewer_H
