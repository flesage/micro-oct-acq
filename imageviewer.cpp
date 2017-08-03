#include "imageviewer.h"
#include "config.h"
#include <QMatrix>
#include <QTime>
#include <iostream>
#include <cmath>

ImageViewer::ImageViewer(QWidget *parent, int n_alines, float fwhm, float line_period) :
    QLabel(parent), p_n_alines(n_alines)
{
    is_fringe_mode = true;
    is_focus_line = false;
    p_threshold =0.001;
    f_fft.init(LINE_ARRAY_SIZE,p_n_alines);
    f_fft.init_doppler(fwhm,line_period);
    p_fringe_image = QImage(LINE_ARRAY_SIZE,p_n_alines,QImage::Format_Indexed8);
    p_image = QImage(LINE_ARRAY_SIZE/2,p_n_alines,QImage::Format_Indexed8);

    pix = QPixmap::fromImage(p_fringe_image);
    setPixmap(pix);
    p_dimage = (double*) new double[p_n_alines*LINE_ARRAY_SIZE];
    p_data_buffer=new unsigned short[p_n_alines*LINE_ARRAY_SIZE];
    p_f_data_buffer=new float[p_n_alines*LINE_ARRAY_SIZE];
    p_doppler_signal=new float[p_n_alines*LINE_ARRAY_SIZE];

    oct_image=new float[(LINE_ARRAY_SIZE/2+1)*p_n_alines];
    setFocusPolicy(Qt::StrongFocus);
    resize(200,400);
    real_fringe = new double[2048];

//    QVector<QRgb> anat_color_table;
//    for(int i = 0; i < 256; ++i)
//    {
//        anat_color_table.append(qRgb(i,0,255-i));
//    }
//    p_image.setColorTable(anat_color_table);
}

ImageViewer::~ImageViewer()
{
    delete [] p_dimage;
    delete [] p_data_buffer;
    delete [] real_fringe;
    delete [] p_f_data_buffer;
    delete [] oct_image;
    delete [] p_doppler_signal;

}

void ImageViewer::updateThreshold(int new_value)
{
    p_threshold = (new_value+1)*0.00001;
}

int ImageViewer::heightForWidth( int width ) const
{
    return ((qreal)pix.height()*width)/pix.width();
}

QSize ImageViewer::sizeHint() const
{
    int w = this->width();
    return QSize( w, heightForWidth(w) );
}

void ImageViewer::setPixmap ( const QPixmap & p)
{
    pix = p;
    QLabel::setPixmap(p);
}

void ImageViewer::resizeEvent(QResizeEvent *)
{
    if(!pix.isNull())
        QLabel::setPixmap(pix.scaled(this->size(),
                                     Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
}

void  ImageViewer::keyPressEvent(QKeyEvent *event)
{
    switch( event->key() )
    {
    case Qt::Key_Space:
    {
        event->accept();
        is_fringe_mode = !is_fringe_mode;
    }
        break;
    case Qt::Key_L:
    {
        event->accept();
        is_focus_line = !is_focus_line;
    }

    default:
        event->ignore();
        break;
    }
}

void ImageViewer::updateView()
{
    // We receive a uint16 image that we need to transform to uint8 for display
    int n_pts = p_n_alines * LINE_ARRAY_SIZE;
    int n_img_pts = p_n_alines * (LINE_ARRAY_SIZE/2+1);
    if(is_fringe_mode)
    {
        unsigned short min = 4096;
        unsigned short max = 0;
        p_mutex.lock();
        for(int i=0;i<n_img_pts;i++)
        {
            if(p_data_buffer[i]>max) max=p_data_buffer[i];
            if(p_data_buffer[i]<min) min=p_data_buffer[i];
        }
        if (max == 0) max = 1;
        // Rescale 12 bits to uint8
        for(int i=0;i<n_pts;i++)
        {
            p_fringe_image.bits()[i]=(unsigned char) ((p_data_buffer[i]-min)*255/(max-min));
        }

        p_mutex.unlock();
        pix = QPixmap::fromImage(p_fringe_image);

    }
    else
    {
        p_mutex.lock();
        f_fft.interp_and_do_fft(p_data_buffer, oct_image);
        p_mutex.unlock();

        f_fft.compute_doppler(p_doppler_signal);
        float min = (float) 1e12;
        float max = (float) -1e12;
        for( int i =0; i< n_img_pts;i++)
        {
            if(oct_image[i]>max) max=oct_image[i];
            if(oct_image[i]<min) min=oct_image[i];
        }
        // Convert to 8 bits
        for(int j=0;j<p_n_alines;j++)
        {
        for(int i=0;i<1024;i++)
        {
            p_image.bits()[i+j*(LINE_ARRAY_SIZE/2)]=(unsigned char) ((oct_image[i+j*(LINE_ARRAY_SIZE/2+1)]-min)*255/(max-min));
        }
        }
        if( is_focus_line )
        {
            for(int i=0;i<p_n_alines;i++) p_image.bits()[i*(LINE_ARRAY_SIZE/2)+130] = 255;
            for(int i=0;i<p_n_alines;i++) p_image.bits()[i*(LINE_ARRAY_SIZE/2)+65] = 255;
        }

        QRect rect(0,0,512,p_n_alines);
        QImage tmp = p_image.copy(rect);
        pix = QPixmap::fromImage(tmp);

    }

    // Set as pixmap
    QMatrix rm;
    rm.rotate(90);
    pix=pix.transformed(rm);
    QLabel::setPixmap(pix.scaled(this->size(),
                                 Qt::IgnoreAspectRatio, Qt::SmoothTransformation));

}

void ImageViewer::put(unsigned short* data)
{
    if (p_mutex.tryLock())
    {
        memcpy(p_data_buffer,data,p_n_alines*2048*sizeof(unsigned short));
        p_mutex.unlock();
    }
}
