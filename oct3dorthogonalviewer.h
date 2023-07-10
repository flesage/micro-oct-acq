#ifndef OCT3DORTHOGONALVIEWER_H
#define OCT3DORTHOGONALVIEWER_H

#include <QWidget>
#include <QString>
#include <QImage>
#include <QPixmap>
#include <QPen>
#include "arrayfire.h"
#include <QMutex>
#include "FringeFFT.h"

namespace Ui {
class oct3dOrthogonalViewer;
}

class oct3dOrthogonalViewer : public QWidget
{
    enum ProjectionMode
    {
        AVERAGE = 0,
        MAXIMUM = 1,
        MINIMUM = 2,
        VARIANCE = 3,
    };

    Q_OBJECT

public:
    explicit oct3dOrthogonalViewer(QWidget *parent = nullptr, int nx=64, int n_extra=40, int ny=64, int nz=64);
    ~oct3dOrthogonalViewer();
    void put(unsigned short* frame, unsigned int frame_number);

public slots:
    void slot_update_view();
    void resizeEvent(QResizeEvent *);
    void set_x(int value);
    void set_y(int value);
    void set_z(int value);
    void set_slice_thickness(int value);
    void set_projection_mode(int type);
    void set_log_transform(int value);
    void set_overlay(int value);

private:
    Ui::oct3dOrthogonalViewer *ui;
    af::array p_oct;
    af::array p_oct_buffer;
    FringeFFT f_fft;
    int p_current_x;
    int p_current_y;
    int p_current_z;
    int p_current_frame;
    float p_line_thickness;
    int p_x_thickness;
    int p_y_thickness;
    int p_z_thickness;
    int p_slice_thickness;
    int p_projection_mode;
    int p_nx;
    int p_ny;
    int p_nz;
    int p_n_extra;
    bool p_log_transform;
    bool p_overlay;
    QPixmap pix_xy;
    QPixmap pix_xz;
    QPixmap pix_yz;
    QImage p_image_xy;
    QImage p_image_xz;
    QImage p_image_yz;
    bool ui_is_ready;
    unsigned short int* p_data_buffer;
    unsigned char* p_image_buffer;
    QMutex p_mutex;

};

#endif // OCT3DORTHOGONALVIEWER_H
