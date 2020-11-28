#ifndef ANGIOVIEWER3DFORM_H
#define ANGIOVIEWER3DFORM_H

#include <QWidget>
#include <QMutex>
#include <QImage>
#include <QPixmap>
#include <QMouseEvent>
#include <QEvent>
#include "arrayfire.h"

namespace Ui {
class AngioViewer3DForm;
}

class AngioViewer3DForm : public QWidget
{
    Q_OBJECT

public:
    explicit AngioViewer3DForm(QWidget *parent = nullptr, int nx=512, int n_extra = 40, int ny=512, int nz=512);
    ~AngioViewer3DForm();
    void put(const af::array& angio_data, unsigned int frame_number);
    bool eventFilter( QObject* watched, QEvent* event );

public slots:
    void updateView();
    void changeDepth(int value);
    void changeSliceThickness();
    void resizeEvent(QResizeEvent *);
    void setAverageFlag(bool flag);
signals:
    void sig_updateLineScanPos(int,int,int,int);
private:
    Ui::AngioViewer3DForm *ui;
    af::array p_angio;
    unsigned char* p_current_slice;
    unsigned char* p_tmp_avg;
    unsigned int p_current_frame;
    QImage p_image;
    QPixmap pix;
    int p_current_depth;
    int p_nx;
    int p_ny;
    int p_nz;
    int p_slice_thickness;
    int* p_average;
    int p_offset;
    int p_start_x;
    int p_start_y;
    int p_stop_x;
    int p_stop_y;
    bool p_show_line;
    bool p_average_on;
};

#endif // ANGIOVIEWER3DFORM_H
