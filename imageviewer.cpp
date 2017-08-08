#include "imageviewer.h"
#include "config.h"
#include <QMatrix>
#include <QTime>
#include <iostream>
#include <cmath>

ImageViewer::ImageViewer(QWidget *parent, int n_alines, float msec_fwhm, float line_period, float spatial_fwhm, float dimz, float dimx) :
    QLabel(parent), p_n_alines(n_alines)
{
    is_fringe_mode = true;
    is_focus_line = false;
    is_doppler = false;
    p_threshold =0.1;
    f_fft.init(LINE_ARRAY_SIZE,p_n_alines,dimz, dimx);
    f_fft.init_doppler(msec_fwhm,line_period,spatial_fwhm);
    p_fringe_image = QImage(LINE_ARRAY_SIZE,p_n_alines,QImage::Format_Indexed8);
    p_image = QImage(LINE_ARRAY_SIZE/2+1,p_n_alines,QImage::Format_Indexed8);
    p_doppler_image = QImage(LINE_ARRAY_SIZE/2+1,p_n_alines-1,QImage::Format_Indexed8);

    pix = QPixmap::fromImage(p_fringe_image);
    setPixmap(pix);
    p_data_buffer=new unsigned short[p_n_alines*LINE_ARRAY_SIZE];

    setFocusPolicy(Qt::StrongFocus);
    resize(200,400);

    QVector<QRgb> dop_color_table;
    for(int i = 0; i < 128; ++i)
    {
        dop_color_table.append(qRgb(2*i,2*i,255));
    }
    for(int i = 128; i < 255; ++i)
    {
        dop_color_table.append(qRgb(255,511-2*i,511-2*i));
    }
    p_doppler_image.setColorTable(dop_color_table);
}

ImageViewer::~ImageViewer()
{
    delete [] p_data_buffer;
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
        break;
    case Qt::Key_D:
    {
        event->accept();
        is_doppler = !is_doppler;
    }
        break;
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
        QMatrix rm;
        rm.rotate(90);
        pix=pix.transformed(rm);
    }
    else
    {

        QImage tmp;
        if(is_doppler)
        {
            p_mutex.lock();
            f_fft.compute_doppler(p_data_buffer,p_doppler_image.bits());
            p_mutex.unlock();
            QRect rect(0,0,512,p_n_alines);
            tmp = p_doppler_image.copy(rect);
        }
        else
        {
            p_mutex.lock();
            f_fft.interp_and_do_fft(p_data_buffer, p_image.bits(),p_threshold);
            p_mutex.unlock();
            QRect rect(0,0,512,p_n_alines);
            tmp = p_image.copy(rect);
        }
        pix = QPixmap::fromImage(tmp);
    }

    // Set as pixmap
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
