#include <iostream>
#include <QPainter>
#include <QPen>
#include "angioviewer3dform.h"
#include "ui_angioviewer3dform.h"

AngioViewer3DForm::AngioViewer3DForm(QWidget *parent,int nx, int ny, int nz) :
    QWidget(parent),
    ui(new Ui::AngioViewer3DForm), p_nx(nx), p_ny(ny), p_nz(nz), p_slice_thickness(5)
{
    ui->setupUi(this);
    ui->horizontalSlider_zpos->setRange(0,p_nz-1);
    p_angio=af::array(p_nz,p_nx,p_ny,f32);
    p_current_slice = new unsigned char[p_nx*p_ny];
    p_average = new int[ny];
    p_tmp_avg = new unsigned char[p_nx*p_nz];
    p_image = QImage(p_nx,p_ny,QImage::Format_Indexed8);
    p_current_frame=0;
    p_current_depth=0;
    for(int i=0;i<p_ny;i++) p_average[i]=0;
    connect(ui->horizontalSlider_zpos, &QSlider::valueChanged, this, &AngioViewer3DForm::changeDepth );
    connect(ui->lineEdit_sliceThickness,SIGNAL(returnPressed()),this,SLOT(changeSliceThickness()));
    ui->label_angioview->installEventFilter( this );
    p_start_x=0;
    p_stop_x=0;
    p_start_y=0;
    p_stop_y=0;
    p_show_line = true;
}

AngioViewer3DForm::~AngioViewer3DForm()
{
    delete [] p_current_slice;
    delete [] p_average;
    delete [] p_tmp_avg;
    delete ui;
}

bool AngioViewer3DForm::eventFilter( QObject* watched, QEvent* event ) {
    if ( watched != ui->label_angioview )
        return false;
    if ( event->type() != QEvent::MouseButtonPress && event->type() != QEvent::MouseButtonRelease
         && event->type() != QEvent::MouseMove ) return false;
    const QMouseEvent* const me = static_cast<const QMouseEvent*>( event );


    //might want to check the buttons here
    if ( event->type() == QEvent::MouseButtonPress)
    {
        p_start_x=(int) (1.0*me->x()/ui->label_angioview->size().width()*p_nx);
        p_start_y=(int) (1.0*me->y()/ui->label_angioview->size().height()*p_ny);
        p_stop_x = p_start_x;
        p_stop_y = p_start_y;
    }
    if ( event->type() == QEvent::MouseMove)
    {
        if( p_show_line )
        {
            p_stop_x=(int) (1.0*me->x()/ui->label_angioview->size().width()*p_nx);
            p_stop_y=(int) (1.0*me->y()/ui->label_angioview->size().height()*p_ny);
        }
    }

    if ( event->type() == QEvent::MouseButtonRelease)
    {
        p_stop_x=(int) (1.0*me->x()/ui->label_angioview->size().width()*p_nx);
        p_stop_y=(int) (1.0*me->y()/ui->label_angioview->size().height()*p_ny);
        emit sig_updateLineScanPos(p_start_x,p_start_y,p_stop_x,p_stop_y);
    }
    updateView();
    return false;
}


void AngioViewer3DForm::put(const af::array& data, unsigned int frame_number)
{
    p_current_frame = (frame_number)%p_ny;
    p_average[p_current_frame]+=1;
    // Copy current angio
    p_angio(af::span,af::span,p_current_frame)=data/p_average[p_current_frame]+
            p_angio(af::span,af::span,p_current_frame)*(p_average[p_current_frame]-1.0)/p_average[p_current_frame];

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
    // Take MIP
    int n_slices;
    if(p_current_depth+p_slice_thickness>p_nz-1)
    {
        n_slices = p_nz-p_current_depth;
    }
    else
    {
        n_slices=p_slice_thickness;
    }
    // Now filter the mip
    af::array g = af::gaussianKernel(3, 3, 0, 0);
    af::array mip=af::mean(p_angio(af::seq(p_current_depth,p_current_depth+n_slices-1),af::span,af::span),0);
    mip=af::convolve2(mip,g);
    //af::array mip=af::convolve(g,af::mean(p_angio(af::seq(p_current_depth,p_current_depth+n_slices-1),af::span,af::span),0));
    float l_max = af::max<float>(mip);
    float l_min = af::min<float>(mip);
    mip=255.0*(mip-l_min)/(l_max-l_min);
    mip.as(u8).host(p_image.bits());
    QImage tmp = p_image.convertToFormat(QImage::Format_ARGB32);

    pix = QPixmap::fromImage(tmp);
    QPainter painter(&pix);
    QPen pen(Qt::red);
    painter.drawLine(p_start_x,p_start_y,p_stop_x, p_stop_y);
    // Set as pixmap
    ui->label_angioview->setPixmap(pix.scaled(ui->label_angioview->size(),
                                              Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
}


void AngioViewer3DForm::resizeEvent(QResizeEvent *)
{
    if(!pix.isNull())
       ui->label_angioview->setPixmap(pix.scaled(ui->label_angioview->size(),
                                     Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
}
