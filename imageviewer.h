#ifndef ImageViewer_H
#define ImageViewer_H

#include <QImage>
#include <QLabel>
#include <QKeyEvent>
#include <fftw3.h>

class QScrollBar;

class ImageViewer : public QLabel
{
    Q_OBJECT
public:
    explicit ImageViewer(QWidget *parent = 0, int n_alines = 100);
    virtual ~ImageViewer();
    virtual int heightForWidth( int width ) const;
    virtual QSize sizeHint() const;
    void put(unsigned short* frame);
signals:

public slots:
    void updateView();
    void resizeEvent(QResizeEvent *);
    void setPixmap ( const QPixmap & );
    void updateThreshold(int);
protected:
    virtual void  keyPressEvent(QKeyEvent *event);

private:

    double scaleFactor;
    int p_n_alines;
    QImage p_image;
    QImage p_fringe_image;

    double* p_dimage;
    double p_threshold;
    unsigned short int* p_data_buffer;
    QMutex p_mutex;
    bool is_fringe_mode;
    bool is_focus_line;
    QPixmap pix;
    fftw_complex* oct_image;
    double* real_fringe;
    fftw_plan fft_plan;
    double* p_interpolation_matrix;
    int* p_band_start;
    int* p_band_stop;

};

#endif // ImageViewer_H
