#ifndef OCT3DORTHOGONALVIEWER_H
#define OCT3DORTHOGONALVIEWER_H

#include <QWidget>
#include <QString>
#include <QImage>
#include <QPixmap>
#include "arrayfire.h"

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
    explicit oct3dOrthogonalViewer(QWidget *parent = nullptr);
    ~oct3dOrthogonalViewer();

public slots:
    void slot_update_view();
    //void resizeEvent(QResizeEvent *);
    void set_x(int value);
    void set_y(int value);
    void set_z(int value);
    void set_slice_thickness(int value);
    void set_projection_mode(int type);

private:
    Ui::oct3dOrthogonalViewer *ui;
    af::array p_oct;
    int p_current_x;
    int p_current_y;
    int p_current_z;
    int p_slice_thickness;
    int p_projection_mode;
    int p_nx;
    int p_ny;
    int p_nz;
    QImage p_image_xy;
    QPixmap pix;
    QImage p_image_xz;
    QImage p_image_yz;
    bool ui_is_ready;
};

#endif // OCT3DORTHOGONALVIEWER_H
