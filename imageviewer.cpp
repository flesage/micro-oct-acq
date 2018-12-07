#include "imageviewer.h"
#include "config.h"
#include <QMatrix>
#include <QTime>
#include <iostream>
#include <cmath>

ImageViewer::ImageViewer(QWidget *parent, int n_alines, float msec_fwhm, float line_period, float spatial_fwhm, float dimz, float dimx) :
    QLabel(parent), p_n_alines(n_alines)
{
    p_current_viewmode = FRINGE;
    p_view_depth=512;
    is_focus_line = false;
    is_optimization = false;
    p_image_threshold =0.001f;
    p_hanning_threshold = 1e-6f;
    f_fft.init(LINE_ARRAY_SIZE,p_n_alines,dimz, dimx);
    f_fft.init_doppler(msec_fwhm,line_period,spatial_fwhm);
    p_fringe_image = QImage(LINE_ARRAY_SIZE,p_n_alines,QImage::Format_Indexed8);
    p_image = QImage(LINE_ARRAY_SIZE/2,p_n_alines,QImage::Format_Indexed8);
    p_hilbert_image = QImage(LINE_ARRAY_SIZE,p_n_alines,QImage::Format_Indexed8);
    p_doppler_image = QImage(LINE_ARRAY_SIZE/2,p_n_alines-1,QImage::Format_Indexed8);
    p_fwhm_view = new FWHMViewer(0,p_view_depth);
    p_phase_view = new FWHMViewer(0,2048);

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
    p_fwhm_view->close();
    p_phase_view->close();
    delete p_fwhm_view;
    delete p_phase_view;
}

void ImageViewer::updateImageThreshold(float new_value)
{
    p_image_threshold = new_value;
}

void ImageViewer::updateHanningThreshold(float new_value)
{
    p_hanning_threshold = new_value;
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
    case Qt::Key_D:
    {
        event->accept();
        p_current_viewmode=DOPPLER;
    }
        break;
    case Qt::Key_F:
    {
        event->accept();
        p_current_viewmode=FRINGE;
    }
        break;
    case Qt::Key_H:
    {
        event->accept();
        p_current_viewmode=HILBERT;
    }
        break;
    case Qt::Key_L:
    {
        event->accept();
        is_focus_line = !is_focus_line;
    }
        break;
    case Qt::Key_O:
    {
        event->accept();
        is_optimization = !is_optimization;
        if( p_current_viewmode==STRUCT )
        {
            if(is_optimization)
            {
                p_fwhm_view->show();
            }
            else
            {
                p_fwhm_view->hide();
            }
        }
        if( p_current_viewmode==HILBERT )
        {
            if(is_optimization)
            {
                p_phase_view->show();
            }
            else
            {
                p_phase_view->hide();
            }
        }
    }
        break;
    case Qt::Key_S:
    {
        event->accept();
        p_current_viewmode=STRUCT;
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
    QImage tmp;
    QMatrix rm;
    rm.rotate(90);
    QRect rect;

    switch(p_current_viewmode)
    {
    case FRINGE:
    {
        unsigned short min = 4096;
        unsigned short max = 0;
        p_mutex.lock();
        for(int i=0;i<n_pts;i++)
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
        pix=pix.transformed(rm);
    }
        break;
    case DOPPLER:
    {
        p_mutex.lock();
        f_fft.compute_doppler(p_data_buffer,p_doppler_image.bits(),p_image_threshold, p_hanning_threshold);
        p_mutex.unlock();
        rect.setRect(0,0,512,p_n_alines-1);
        tmp = p_doppler_image.copy(rect);
        pix = QPixmap::fromImage(tmp);
        pix=pix.transformed(rm);
    }
        break;
    case HILBERT:
        p_mutex.lock();
        f_fft.compute_hilbert(p_data_buffer,p_hilbert_image.bits(), p_hanning_threshold);
        p_mutex.unlock();
        rect.setRect(0,0,2048,p_n_alines);
        tmp = p_hilbert_image.copy(rect);
        pix = QPixmap::fromImage(tmp);
        pix=pix.transformed(rm);
        if(is_optimization)
        {
            // Push middle line
            p_phase_view->put(&p_hilbert_image.bits()[p_n_alines/2*2048]);
        }
        break;
    case STRUCT:
    {
        p_mutex.lock();
        f_fft.interp_and_do_fft(p_data_buffer, p_image.bits(),p_image_threshold, p_hanning_threshold);
        p_mutex.unlock();
        rect.setRect(0,0,p_view_depth,p_n_alines);
        tmp = p_image.copy(rect);
        pix = QPixmap::fromImage(tmp);
        pix=pix.transformed(rm);
        if(is_optimization)
        {
            // Push middle line
            p_fwhm_view->put(&p_image.bits()[p_n_alines/2*1024]);
        }
    }
        break;

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
