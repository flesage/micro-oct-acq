#ifndef OCT3DORTHOGONALVIEWER_H
#define OCT3DORTHOGONALVIEWER_H

#include <QWidget>
#include <QString>
#include <QImage>
#include <QPixmap>
#include <QPen>
#include "arrayfire.h"
#include <QMutex>

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
    explicit oct3dOrthogonalViewer(QWidget *parent = nullptr, int nx=64, int ny=64, int nz=64);
    ~oct3dOrthogonalViewer();
    void put(unsigned short* frame, unsigned int frame_number);
    //void put(const af::array& oct_data, unsigned int frame_number); // TODO: method to update the OCT data
    void reconstruct(unsigned short* in_fringe, unsigned short* out_image);

    //void
    // bool eventFilter(QObject* watched, QEvent* event);

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
    void simulate_bscan();

private:
    Ui::oct3dOrthogonalViewer *ui;
    af::array p_oct;
    int p_current_x;
    int p_current_y;
    int p_current_z;
    int p_line_thickness;
    int p_slice_thickness;
    int p_projection_mode;
    int p_nx;
    int p_ny;
    int p_nz;
    bool p_log_transform;
    bool p_overlay;
    QPixmap pix_xy;
    QPixmap pix_xz;
    QPixmap pix_yz;
    QImage p_image_xy;
    QImage p_image_xz;
    QImage p_image_yz;
    QPen pen_x;
    QPen pen_y;
    QPen pen_z;
    bool ui_is_ready;
    unsigned short int* p_data_buffer;
    unsigned short int* p_image_buffer;
    QMutex p_mutex;

    // Simulation
    QTimer* simulation_timer;
    af::array simulated_bscan;
    int p_y_sim;

};

#endif // OCT3DORTHOGONALVIEWER_H
