#ifndef ANGIOVIEWER3DFORM_H
#define ANGIOVIEWER3DFORM_H

#include <QWidget>
#include <QMutex>
#include <QImage>
#include <QPixmap>

namespace Ui {
class AngioViewer3DForm;
}

class AngioViewer3DForm : public QWidget
{
    Q_OBJECT

public:
    explicit AngioViewer3DForm(QWidget *parent = nullptr, int nx=512, int ny=512, int nz=512);
    ~AngioViewer3DForm();
    void put(unsigned char* angio_data);
public slots:
    void updateView();
    void changeDepth(int value);

private:
    Ui::AngioViewer3DForm *ui;
    unsigned char* p_angio;
    unsigned char* p_current_slice;
    unsigned int p_current_frame;
    QImage p_image;
    QPixmap pix;
    int p_current_depth;
    int p_nx;
    int p_ny;
    int p_nz;
    QMutex p_mutex;
};

#endif // ANGIOVIEWER3DFORM_H
