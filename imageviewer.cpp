#include "imageviewer.h"
#include "config.h"
#include <QMatrix>
#include <QTime>
#include <iostream>
#include <math.h>

ImageViewer::ImageViewer(QWidget *parent, int n_alines) :
    QLabel(parent), p_n_alines(n_alines)
{
    is_fringe_mode = true;
    is_focus_line = false;
    p_threshold =0.1;
    p_fringe_image = QImage(LINE_ARRAY_SIZE,p_n_alines,QImage::Format_Indexed8);
    p_image = QImage(LINE_ARRAY_SIZE/2,p_n_alines,QImage::Format_Indexed8);

    pix = QPixmap::fromImage(p_fringe_image);
    setPixmap(pix);
    p_dimage = (double*) new double[p_n_alines*LINE_ARRAY_SIZE];
    p_data_buffer=new unsigned short[p_n_alines*LINE_ARRAY_SIZE];
    setFocusPolicy(Qt::StrongFocus);
    resize(200,400);
    // Prepare fft plan
    real_fringe = new double[2048];
    oct_image = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * 1025);
    fft_plan = fftw_plan_dft_r2c_1d(2048,
                                    real_fringe, oct_image,
                                    FFTW_MEASURE);
    p_interpolation_matrix = new double[2048*2048];
    FILE* fp=fopen("C:\\Users\\Public\\Documents\\interpolation_matrix.dat","rb");
    fread(p_interpolation_matrix,sizeof(double),2048*2048,fp);
    fclose(fp);

    // For interpolation acceleration
    p_band_start = new int[2048];
    p_band_stop = new int[2048];
    for(int i=0;i<2048;i++)
    {
        bool found_start = false;
        p_band_start[i]=0;
        for( int j=0; j<2048;j++)
        {
            if(!found_start && p_interpolation_matrix[i+j*2048]<1e-8)
            {
                p_band_start[i]=p_band_start[i]+1;
            }
            else
            {
                if(!found_start)
                {
                    found_start = true;
                    p_band_start[i]=j;
                }
                else if (p_interpolation_matrix[i+j*2048]<1e-8)
                {
                    p_band_stop[i]=j-1;
                    break;
                }
                else if( j == 2047 )
                {
                    p_band_stop[i]=j;
                }
            }
        }
    }
    p_band_stop[2047] = 2047;
}

ImageViewer::~ImageViewer()
{
    delete [] p_dimage;
    delete [] p_data_buffer;
    fftw_destroy_plan(fft_plan);
    fftw_free(oct_image);
    delete [] real_fringe;
    delete [] p_interpolation_matrix;
    delete [] p_band_start;
    delete [] p_band_stop;
}

void ImageViewer::updateThreshold(int new_value)
{
    p_threshold = (new_value+1)*10.0;
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
//    QTime t;
//    t.start();
    // We receive a uint16 image that we need to transform to uint8 for display
    int n_pts = p_n_alines * LINE_ARRAY_SIZE;

    if(is_fringe_mode)
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

    }
    else
    {
        double avg_fringe[2048];
        p_mutex.lock();
        // Compute avg fringe (could be done less frequently)
        for (int i=0; i<2048; i++)
        {
           avg_fringe[i]=p_data_buffer[i];
        }
        for (int j=1 ; j<p_n_alines; j++)
        {
            for (int i=0; i<2048; i++)
            {
                avg_fringe[i]+=p_data_buffer[j*2048+i];
            }
        }
//        std::cerr << "computed avg" << std::endl;
        // Fill double data substract mean and interpolate
        double max=log(p_threshold);
        double min=1000000.0;

        for (int j=0; j<p_n_alines; j++)
        {
            for (int i=0; i<2048; i++)
            {
                real_fringe[i]=0.0;
                for(int ik=p_band_start[i]; ik<=p_band_stop[i]; ik++)
                {
                    real_fringe[i]+=p_interpolation_matrix[i+ik*2048]*(p_data_buffer[j*2048+ik]-avg_fringe[ik]/p_n_alines);
                }
            }
            // Do FFT for that line
            fftw_execute(fft_plan);
            for(int i=0;i<LINE_ARRAY_SIZE/2;i++)
            {
                int imx=(LINE_ARRAY_SIZE/2)*j+i;
                double tmp = 0.0;
                if (i>10)
                tmp=sqrt(oct_image[i][0]*oct_image[i][0]+oct_image[i][1]*oct_image[i][1]);
                p_dimage[imx]=log(tmp+p_threshold);
                if(p_dimage[imx]>max) max=p_dimage[imx];
                if(p_dimage[imx]<min) min=p_dimage[imx];
            }
        }
        p_mutex.unlock();
//        std::cerr << "computed int" << std::endl;

        // Convert to 8 bits
        for(int i=0;i<n_pts/2;i++)
        {
            p_image.bits()[i]=(unsigned char) ((p_dimage[i]-min)*255/(max-min));
        }
        if( is_focus_line )
        {
            for(int i=0;i<p_n_alines;i++) p_image.bits()[i*LINE_ARRAY_SIZE/2+130] = 255;
            for(int i=0;i<p_n_alines;i++) p_image.bits()[i*LINE_ARRAY_SIZE/2+65] = 255;

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
//    std::cerr << t.elapsed() << "ms" << std::endl;
}

void ImageViewer::put(unsigned short* data)
{
    if (p_mutex.tryLock())
    {
        memcpy(p_data_buffer,data,p_n_alines*2048*sizeof(unsigned short));
        p_mutex.unlock();
    }
}
