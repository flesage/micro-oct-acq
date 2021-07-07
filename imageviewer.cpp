#include "imageviewer.h"
#include "config.h"
#include <QMatrix4x4>
#include <QTime>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <QPainter>
#include "arrayfire.h"

ImageViewer::ImageViewer(QWidget *parent, int n_alines, int n_extra, int ny, int view_depth, unsigned int n_repeat, float msec_fwhm, float line_period, float spatial_fwhm,
                         float dimz, float dimx, int factor, DM* in_dm, float** in_zern, int in_n_act, int in_z_mode_min, int in_z_mode_max ) :
    QLabel(parent), p_ny(ny), p_factor(factor), p_n_repeat(n_repeat), f_fft(n_repeat,factor), p_n_alines(n_alines), p_n_extra(n_extra), p_view_depth(view_depth), dm(in_dm), Z2C(in_zern), nbAct(in_n_act), z_mode_min(in_z_mode_min), z_mode_max(in_z_mode_max)
{
    p_current_viewmode = STRUCT;
    is_focus_line = false;
    is_optimization = false;
    is_dm_optimization = false;
    p_image_threshold =0.001f;
    p_hanning_threshold = 1e-6f;
    f_fft.init(LINE_ARRAY_SIZE,p_n_alines,dimz, dimx);
    f_fft.init_doppler(msec_fwhm,line_period,spatial_fwhm);
    p_fringe_image = QImage(LINE_ARRAY_SIZE,p_n_alines,QImage::Format_Indexed8);
    p_image = QImage(LINE_ARRAY_SIZE/2,p_n_alines,QImage::Format_Indexed8);
    p_hilbert_image = QImage(LINE_ARRAY_SIZE,p_n_alines,QImage::Format_Indexed8);
    p_doppler_image = QImage(LINE_ARRAY_SIZE/2,p_n_alines-1,QImage::Format_Indexed8);
    p_fwhm_view = new FWHMViewer(0,p_view_depth);
    p_phase_view = new FWHMViewer(0,LINE_ARRAY_SIZE);
    p_angio_view = new AngioViewer3DForm(0,n_alines, n_extra, p_ny, p_view_depth);
    connect(p_angio_view,SIGNAL(sig_updateLineScanPos(int,int,int,int)),this,SLOT(updateLineScanPos(int,int,int,int)));
    connect(this,SIGNAL(sig_updateAverageAngio(bool)),p_angio_view,SLOT(setAverageFlag(bool)));

    p_angio_averageFlag = false;
    p_angio_view->setAverageFlag(p_angio_averageFlag);




    p_line_status = false;
    p_start_line = 0;
    p_stop_line = 10;
    p_frame_number = 0;

    pix = QPixmap::fromImage(p_fringe_image);
    setPixmap(pix);
    p_data_buffer=new unsigned short[p_n_alines*LINE_ARRAY_SIZE*factor];

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


    // Initialize dm_data
    dm_data = new Scalar[97];
    current_opt_dm_data = new Scalar[97];
    for (int i = 0; i < nbAct; i++)
    {
        dm_data[i] = 0;
        current_opt_dm_data[i] = 0;
    }

    // Initialize max_metric
    max_metric = 0;

    // Intialize dm_output_file_number
    dm_output_file_number = 0;
}

ImageViewer::~ImageViewer()
{
    delete [] p_data_buffer;
    p_fwhm_view->close();
    p_phase_view->close();
    delete p_fwhm_view;
    delete p_phase_view;
    p_angio_view->close();
    delete p_angio_view;

    delete [] dm_data;
    delete [] current_opt_dm_data;
}

void ImageViewer::optimizeDM(void)
{
    if(!is_dm_optimization)
    {
        // Set dm optimization flag
        is_dm_optimization=true;

        // Initialize z_idx and z_idx_max
        z_idx = z_mode_min*(z_mode_min+1)/2-1;
        z_idx_max = z_mode_max*(z_mode_max+1)/2-1+z_mode_max;

        // Initialize dm_c and dm_c_max
        dm_c = Z2C[z_idx][97];
        dm_c_max = 0;

        // Initialize mirror position
        moveDM(z_idx,dm_c);

        // Open output file
        dm_ouput_file.open("D:/dm_optimization_data/dm_data_" + std::to_string(dm_output_file_number) + ".txt");
        dm_output_file_number++;
    }
}

void ImageViewer::turnDMOn(void)
{
    if(!is_dm_optimization)
    {
        dm->Send(current_opt_dm_data);
    }
}

void ImageViewer::turnDMOff(void)
{
    if(!is_dm_optimization)
    {
        dm->Reset();
    }
}

void ImageViewer::resetDM(void)
{
    if(!is_dm_optimization)
    {
        for(int i=0;i<nbAct;i++) current_opt_dm_data[i]=0;
        max_metric = 0; // Reset max_metric to restart optimization from the beginning
        dm->Reset();
    }
}

void ImageViewer::set_disp_comp_vect(float* disp_comp_vector)
{
    f_fft.set_disp_comp_vect(disp_comp_vector);
}

void ImageViewer::updateLineScanPos(int start_x, int start_y, int stop_x, int stop_y)
{
    emit sig_updateLineScanPos(start_x,start_y,stop_x,stop_y);
}

void ImageViewer::updateAngioAverageFlag(bool new_value)
{
    p_angio_averageFlag = new_value;
    emit sig_updateAverageAngio(p_angio_averageFlag);
}


void ImageViewer::updateImageThreshold(float new_value)
{
    p_image_threshold = new_value;
}

void ImageViewer::updateHanningThreshold(float new_value)
{
    p_hanning_threshold = new_value;
}

void ImageViewer::updateAngioAlgo(int new_value)
{
    p_angio_algo = new_value;
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

void ImageViewer::setCurrentViewModeStruct()
{
    p_current_viewmode=STRUCT;
}

void  ImageViewer::keyPressEvent(QKeyEvent *event)
{
    switch( event->key() )
    {
    case Qt::Key_A:
    {
        event->accept();
        p_angio_view->setAverageFlag(p_angio_averageFlag);
        p_current_viewmode=ANGIO;
        p_fwhm_view->hide();
        p_phase_view->hide();
        if(p_ny > 1)
        {
            p_angio_view->show();
        }
        else
        {
            p_angio_view->hide();
        }
    }
        break;
    case Qt::Key_D:
    {
        event->accept();
        p_current_viewmode=DOPPLER;
        p_fwhm_view->hide();
        p_phase_view->hide();
        p_angio_view->hide();
    }
        break;
    case Qt::Key_F:
    {
        event->accept();
        p_current_viewmode=FRINGE;
        p_fwhm_view->hide();
        p_phase_view->hide();
        p_angio_view->hide();
    }
        break;
    case Qt::Key_H:
    {
        event->accept();
        p_current_viewmode=HILBERT;
        p_fwhm_view->hide();
        p_phase_view->hide();
        p_angio_view->hide();
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
            p_phase_view->hide();
            p_angio_view->hide();
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
            p_fwhm_view->hide();
            p_angio_view->hide();
        }
    }
        break;
    case Qt::Key_S:
    {
        event->accept();
        p_current_viewmode=STRUCT;
        p_fwhm_view->hide();
        p_phase_view->hide();
        p_angio_view->hide();
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
    QTransform transform;
    QTransform trans = transform.rotate(90);
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
        pix = pix.transformed(trans);
    }
        break;
    case DOPPLER:
    {
        p_mutex.lock();
        f_fft.compute_doppler(p_data_buffer,p_doppler_image.bits(),p_image_threshold, p_hanning_threshold);
        p_mutex.unlock();
        rect.setRect(0,0,p_view_depth,p_n_alines-1);
        tmp = p_doppler_image.copy(rect);
        pix = QPixmap::fromImage(tmp);
        pix = pix.transformed(trans);
    }
        break;
    case HILBERT:
        p_mutex.lock();
        f_fft.compute_hilbert(p_data_buffer,p_hilbert_image.bits(), p_hanning_threshold);
        p_mutex.unlock();
        rect.setRect(0,0,2048,p_n_alines);
        tmp = p_hilbert_image.copy(rect);
        pix = QPixmap::fromImage(tmp);
        pix = pix.transformed(trans);
        if(is_optimization)
        {
            // Push middle line
            p_phase_view->put(&p_hilbert_image.bits()[p_n_alines/2*LINE_ARRAY_SIZE]);
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
        pix = pix.transformed(trans);
        if(is_optimization)
        {
            // Push middle line
            p_fwhm_view->put(&p_image.bits()[p_n_alines/2*(LINE_ARRAY_SIZE/2)]);
        }
    }
        break;
    case ANGIO:
    {
        // Here we are insured that p_data_buffer contains n_repeat frame to compute the angio, so every
        // call returns an angio frame. However sometimes if very fast, we could have more than p_n_repeat
        // frames, so we need to loop to not lose frames
        int n_angios = p_factor/p_n_repeat;
        f_fft.setAngioAlgo(p_angio_algo);

        for(int i=0; i<n_angios;i++)
        {
            p_mutex.lock();
            f_fft.get_angio(&p_data_buffer[i*p_n_alines*2048*p_n_repeat], &p_angio ,p_image_threshold, p_hanning_threshold);
            p_mutex.unlock();
            float l_max = af::max<float>(p_angio);
            float l_min = af::min<float>(p_angio);
            p_angio=255.0*(p_angio-l_min)/(l_max-l_min);
            p_angio.as(u8).host(p_image.bits());
            // Push current angio
            rect.setRect(0,0,p_view_depth,p_n_alines);
            tmp = p_image.copy(rect);
            pix = QPixmap::fromImage(tmp);
            pix = pix.transformed(trans);

            if( p_ny > 1 )
            {
                // Here we have to account for potential skips by providing index
                int frame_number = p_frame_number - n_angios + i;
                p_angio_view->put(p_angio, frame_number);
            }
        }
    }
        break;

    }

    if (is_focus_line == true)
    {
        QPainter painter(&pix);
        int Width = 2;
        QImage start_line(p_n_alines,Width,QImage::Format_RGB32);
        start_line.fill(Qt::green);
        QPixmap start_line_pixmap;
        start_line_pixmap=QPixmap::fromImage(start_line);
        painter.drawPixmap(0, p_start_line-1, p_n_alines, Width, start_line_pixmap);
        QImage stop_line(p_n_alines,Width,QImage::Format_RGB32);
        stop_line.fill(Qt::yellow);
        QPixmap stop_line_pixmap;
        stop_line_pixmap=QPixmap::fromImage(stop_line);
        painter.drawPixmap(0, p_stop_line-1, p_n_alines, Width, stop_line_pixmap);
    }

    // Set as pixmap
    QLabel::setPixmap(pix.scaled(this->size(),
                                 Qt::IgnoreAspectRatio, Qt::SmoothTransformation));

    // Optimize DM
    if ((is_dm_optimization && p_current_viewmode == STRUCT) || (is_dm_optimization && p_current_viewmode == ANGIO))
    {
        optimizeDM(p_image);
    }
}

double ImageViewer::getMetric(QImage image, int metric_number)
{
    double metric = 0;

    switch(metric_number)
    {
    case 1: // Summed intensity
    {
        for (int i = p_n_extra; i < p_n_alines; i++)
        {
            for (int k = p_start_line+1024*i; k < p_stop_line+1024*i; k++)
            {
                metric += image.bits()[k];
            }
        }

        break;
    }
    case 2: // Summed intensity of 10% highest pixels
    {
        unsigned int idx = 0;
        int end_idx = (p_n_alines-p_n_extra)*(p_stop_line-p_start_line);
        unsigned char* data_vec = new unsigned char[end_idx];

        for (int i = p_n_extra; i < p_n_alines; i++)
        {
            for (int k = p_start_line+1024*i; k < p_stop_line+1024*i; k++)
            {
                data_vec[idx] = image.bits()[k];
                idx++;
            }
        }

        std::sort(data_vec,&data_vec[end_idx]);

        for (int i = 1; i <= round(end_idx*0.1); i++)
        {
            metric += data_vec[end_idx-i];
        }

        delete [] data_vec;
        break;
    }
    case 3: // Image variance
    {
        double mean = 0;
        int size = 0;
        for (int i = p_n_extra; i < p_n_alines; i++)
        {
            for (int k = p_start_line+1024*i; k < p_stop_line+1024*i; k++)
            {
                mean += image.bits()[k];
                size++;
            }
        }

        mean /= size;

        for (int i = p_n_extra; i < p_n_alines; i++)
        {
            for (int k = p_start_line+1024*i; k < p_stop_line+1024*i; k++)
            {
                metric += pow(image.bits()[k]-mean,2);
            }
        }

        metric /= size;
        break;
    }
    }

    return metric;
}

void ImageViewer::moveDM(int z_poly, double amp)
{
    if (dm->Check())
    {
        for (int i = 0; i < nbAct; i++)
        {
            dm_data[i] = amp*Z2C[z_poly][i]+current_opt_dm_data[i];
        }

        dm->Send(dm_data);
    }
}

void ImageViewer::moveDMandCurrentOpt(int z_poly, double amp)
{
    if (dm->Check())
    {
        for (int i = 0; i < nbAct; i++)
        {
            current_opt_dm_data[i] = amp*Z2C[z_poly][i]+current_opt_dm_data[i];
        }

        dm->Send(current_opt_dm_data);
    }
}

void ImageViewer::optimizeDM(QImage image)
{
    double metric = getMetric(image,3);

    if (z_idx <= z_idx_max)
    {
        if (metric > max_metric && dm_c > Z2C[z_idx][97]+(Z2C[z_idx][98]-Z2C[z_idx][97])/40*4)
        {
            dm_c_max = dm_c;
            max_metric = metric;
        }

        if (dm_c > Z2C[z_idx][98])
        {
            dm_ouput_file << z_idx << " " << dm_c_max << " " << max_metric << std::endl;
            moveDMandCurrentOpt(z_idx, dm_c_max);
            z_idx++;
            dm_c = Z2C[z_idx][97];
            dm_c_max = 0;
        } else
        {
            dm_ouput_file << z_idx << " " << dm_c << " " << metric << std::endl;
            dm_c += (Z2C[z_idx][98]-Z2C[z_idx][97])/40;
            moveDM(z_idx, dm_c);
        }
    }
    else
    {
        // If we get here we finished optimization.
        is_dm_optimization=false;
        dm_ouput_file.close();
    }
}

void ImageViewer::put(unsigned short* data)
{
    p_frame_number+= p_factor/p_n_repeat;
    if (p_mutex.tryLock())
    {
        memcpy(p_data_buffer,data,p_n_alines*2048*sizeof(unsigned short)*p_factor);
        p_mutex.unlock();
        // New Frame, call function to evaluate sharpness or other metric.
        // Then optimize Mirrors
    }
}

void ImageViewer::checkLine(bool lineStatus, int startLine, int stopLine)
{
    p_line_status=lineStatus;
    p_start_line=startLine;
    p_stop_line=stopLine;
}

void ImageViewer::updateViewLinePositions(bool lineStatus, int startLine, int stopLine)
{
    p_line_status=lineStatus;
    p_start_line=startLine;
    p_stop_line=stopLine;
}
