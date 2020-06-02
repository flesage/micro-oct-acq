#include <iostream>
#include "angioviewer3dform.h"
#include "ui_angioviewer3dform.h"

AngioViewer3DForm::AngioViewer3DForm(QWidget *parent,int nx, int ny, int nz) :
    QWidget(parent),
    ui(new Ui::AngioViewer3DForm), p_nx(nx), p_ny(ny), p_nz(nz), p_slice_thickness(5)
{
    ui->setupUi(this);
    ui->horizontalSlider_zpos->setRange(0,p_nz-1);
    p_angio=new unsigned char[p_nx*p_ny*p_nz];
    p_current_slice = new unsigned char[p_nx*p_ny];
    p_image = QImage(p_nx,p_ny,QImage::Format_Indexed8);
    p_current_frame=0;
    p_current_depth=0;
    connect(ui->horizontalSlider_zpos, &QSlider::valueChanged, this, &AngioViewer3DForm::changeDepth );
    connect(ui->lineEdit_sliceThickness,SIGNAL(returnPressed()),this,SLOT(changeSliceThickness()));
}

AngioViewer3DForm::~AngioViewer3DForm()
{
    delete [] p_current_slice;
    delete [] p_angio;
    delete ui;
}

void AngioViewer3DForm::put(unsigned char* data, unsigned int frame_number)
{
    // Could be done more intelligently to average repeats over multiple volumes
    // Look at this later

    p_current_frame = (frame_number-1)%p_ny;
    memcpy(&p_angio[p_current_frame*p_nz*p_nx],data,p_nx*p_nz*sizeof(unsigned char));
    updateView();
}

void AngioViewer3DForm::changeDepth(int depth)
{
    p_current_depth = depth;
}

void AngioViewer3DForm::changeSliceThickness()
{
    p_slice_thickness=ui->lineEdit_sliceThickness->text().toInt();
}

void AngioViewer3DForm::updateView()
{
    // Critical section
    int n_slices;
    if(p_current_depth+p_slice_thickness>p_nz-1)
    {
        n_slices = p_nz-p_current_depth;
    }
    else
    {
        n_slices=p_slice_thickness;
    }

    for(int j =0; j<p_ny; j++){
        for(int i=0; i<p_nx; i++){
            float mip=0.0;
            for(int k=0;k<n_slices;k++)
            {
                mip+=p_angio[p_current_depth+k+p_nz*i+p_nz*p_nx*j];
            }
            p_current_slice[i+p_nx*j]=(unsigned char) (mip/n_slices);
        }
    }
    memcpy(p_image.bits(),p_current_slice,p_nx*p_ny*sizeof(unsigned char));
    pix = QPixmap::fromImage(p_image);
    // Set as pixmap
    ui->label_angioview->setPixmap(pix);
}

