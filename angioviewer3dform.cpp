#include <iostream>
#include "angioviewer3dform.h"
#include "ui_angioviewer3dform.h"

AngioViewer3DForm::AngioViewer3DForm(QWidget *parent,int nx, int ny, int nz) :
    QWidget(parent),
    ui(new Ui::AngioViewer3DForm), p_nx(nx), p_ny(ny), p_nz(nz)
{
    ui->setupUi(this);
    ui->horizontalSlider_zpos->setRange(0,p_nz-1);
    p_angio=new unsigned char[p_nx*p_ny*p_nz];
    p_current_slice = new unsigned char[p_nx*p_ny];
    p_image = QImage(p_nx,p_ny,QImage::Format_Indexed8);
    p_current_frame=0;
    p_current_depth=0;
    connect(ui->horizontalSlider_zpos, &QSlider::valueChanged, this, &AngioViewer3DForm::changeDepth );
}

AngioViewer3DForm::~AngioViewer3DForm()
{
    delete [] p_current_slice;
    delete [] p_angio;
    delete ui;
}

void AngioViewer3DForm::put(unsigned char* data)
{
    if(p_mutex.tryLock())
    {
        // Could be done more intelligently to add repeats over multiple volumes
        // Look at this later
        memcpy(&p_angio[p_current_frame*p_nz*p_nx],data,p_nx*p_nz*sizeof(unsigned char));
        p_current_frame = (p_current_frame+1)%p_ny;
        p_mutex.unlock();
        updateView();
    }
}

void AngioViewer3DForm::changeDepth(int depth)
{
    p_current_depth = depth;
}

void AngioViewer3DForm::updateView()
{
    // Critical section
    p_mutex.lock();
    for(int j =0; j<p_ny; j++){
        for(int i=0; i<p_nx; i++){
            p_current_slice[i+p_nx*j]=p_angio[p_current_depth+p_nz*i+p_nz*p_nx*j];
        }
    }
    p_mutex.unlock();
    memcpy(p_image.bits(),p_current_slice,p_nx*p_ny*sizeof(unsigned char));
    pix = QPixmap::fromImage(p_image);
    // Set as pixmap
    ui->label_angioview->setPixmap(pix);
}

