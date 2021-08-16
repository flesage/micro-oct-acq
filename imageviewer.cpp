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
                         float dimz, float dimx, int factor, DM* in_dm, float** in_zern, int in_z_min, int in_z_max ) :
    QLabel(parent), p_n_alines(n_alines), p_ny(ny), p_factor(factor), p_n_repeat(n_repeat), p_n_extra(n_extra), f_fft(n_repeat,factor), p_view_depth(view_depth), dm(in_dm), Z2C(in_zern), z_min(in_z_min), z_max(in_z_max)
{
    p_current_viewmode = STRUCT;
    is_focus_line = false;
    is_optimization = false;
    is_dm_optimization = false;
    p_image_threshold = 0.001f;
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

    dm_data = new Scalar[97];
    dm_current_opt = new Scalar[97];
    for (int i = 0; i < 97; i++)
    {
        dm_data[i] = 0;
        dm_current_opt[i] = 0;
    }
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
    delete [] dm_current_opt;
}

void ImageViewer::setMetric(int new_metric)
{
    p_metric = new_metric;
}

void ImageViewer::setPercent(double new_percent)
{
    if(new_percent < 1.0) new_percent = 1.0;
    if(new_percent > 100.0) new_percent = 100.0;
    p_percent = new_percent/100.0;
}

void ImageViewer::optimizeDM(void)
{
    if(!is_dm_optimization)
    {
        is_dm_optimization = true;

        z_idx = z_min;
        dm_idx = 0;
        dm_idx_max = 0;
        dm_metric_max = getMetric(p_image,p_metric);
        for(int i = 0; i < 50; i++)
        {
            dm_c[i] = Z2C[z_idx][97]+(Z2C[z_idx][98]-Z2C[z_idx][97])/50*i;
            dm_metric[i] = 0;
        }

        moveDM(dm_data,dm_c[dm_idx]);
    }
}

void ImageViewer::turnDMOn(void)
{
    if(!is_dm_optimization)
    {
        dm->Send(dm_current_opt);
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
        for(int i = 0; i < 97; i++) dm_current_opt[i] = 0;
        dm_metric_max = 0;
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

    if ((is_dm_optimization && p_current_viewmode == STRUCT) || (is_dm_optimization && p_current_viewmode == ANGIO))
    {
        optimizeDM(p_image);
    }
}

double ImageViewer::getMetric(QImage image, int metric_number)
{
    double metric = 0;
    unsigned int idx = 0;
    int end_idx = (p_n_alines-p_n_extra)*(p_stop_line-p_start_line);
    unsigned char* data_vec = new unsigned char[end_idx];

    switch(metric_number)
    {
    case 0: // Intensity
    {
        for (int i = p_n_extra; i < p_n_alines; i++)
        {
            for (int k = p_start_line+1024*i; k < p_stop_line+1024*i; k++)
            {
                data_vec[idx] = image.bits()[k];
                idx++;
            }
        }

        std::sort(data_vec,&data_vec[end_idx]);

        for (int i = 1; i <= round(end_idx*p_percent); i++)
        {
            metric += data_vec[end_idx-i];
        }

        break;
    }
    case 1: // Variance
    {
        double mean = 0;

        for (int i = p_n_extra; i < p_n_alines; i++)
        {
            for (int k = p_start_line+1024*i; k < p_stop_line+1024*i; k++)
            {
                data_vec[idx]= image.bits()[k];
                mean += data_vec[idx];
                idx++;
            }
        }

        mean /= end_idx;

        for (int i = 0; i < end_idx; i++)
        {
            metric += pow(data_vec[i]-mean,2);
        }

        metric /= end_idx;
        break;
    }
    case 2: // Entropy
    {
        int nbBins = round(sqrt(end_idx));

        for (int i = p_n_extra; i < p_n_alines; i++)
        {
            for (int k = p_start_line+1024*i; k < p_stop_line+1024*i; k++)
            {
                data_vec[idx] = image.bits()[k];
                idx++;
            }
        }

        std::sort(data_vec,&data_vec[end_idx]);

        double *bin = new double[nbBins];

        for(int i = 0; i < nbBins; i++)
        {
            bin[i] = 0.0;
            for (int j = 0; j < end_idx; j++)
            {
                unsigned char min = data_vec[0]+((data_vec[end_idx-1]-data_vec[0])/nbBins)*i;
                unsigned char max = data_vec[0]+((data_vec[end_idx-1]-data_vec[0])/nbBins)*(i+1);
                if((data_vec[j] >= min) && (data_vec[j] < max)) bin[i]++;
            }
            if(bin[i] > 0) metric -= (bin[i]/end_idx)*log2(bin[i]/end_idx);
        }

        delete[] bin;
        break;
    }
    }

    delete [] data_vec;
    return metric;
}

void ImageViewer::moveDM(Scalar* data, double amp)
{
    if (dm->Check())
    {
        for (int i = 0; i < 97; i++)
        {
            data[i] = amp*Z2C[z_idx][i]+dm_current_opt[i];
        }

        dm->Send(data);
    }
}

double ImageViewer::polyfit()
{
    double max_coeff = 0;

    if (dm_idx_max != 0)
    {
        double sumX[5];
        for(int i = 0; i < 5; i++)
        {
            sumX[i] = 0;
            for(int j = dm_idx_max-3; j < dm_idx_max+3; j++)
            {
                sumX[i] += pow(dm_c[j],(float)i);
            }
        }

        double sumY[3];
        for(int i = 0; i < 3; i++)
        {
            sumY[i] = 0;
            for(int j = dm_idx_max-3; j < dm_idx_max+3; j++)
            {
                sumY[i] += pow(dm_c[j],(float)i)*dm_metric[j];
            }
        }

        double M[3][3] = {
            {sumX[0], sumX[1], sumX[2]},
            {sumX[1], sumX[2], sumX[3]},
            {sumX[2], sumX[3], sumX[4]}
        };
        double M0[3][3] = {
            {sumY[0], sumX[1], sumX[2]},
            {sumY[1], sumX[2], sumX[3]},
            {sumY[2], sumX[3], sumX[4]}
        };
        double M1[3][3] = {
            {sumX[0], sumY[0], sumX[2]},
            {sumX[1], sumY[1], sumX[3]},
            {sumX[2], sumY[2], sumX[4]}
        };
        double M2[3][3] = {
            {sumX[0], sumX[1], sumY[0]},
            {sumX[1], sumX[2], sumY[1]},
            {sumX[2], sumX[3], sumY[2]}
        };

        double detM = M[0][0]*M[1][1]*M[2][2] + M[0][1]*M[1][2]*M[2][0] + M[0][2]*M[1][0]*M[2][1] - M[0][2]*M[1][1]*M[2][0] - M[0][1]*M[1][0]*M[2][2] - M[0][0]*M[1][2]*M[2][1];
        double detM0 = M0[0][0]*M0[1][1]*M0[2][2] + M0[0][1]*M0[1][2]*M0[2][0] + M0[0][2]*M0[1][0]*M0[2][1] - M0[0][2]*M0[1][1]*M0[2][0] - M0[0][1]*M0[1][0]*M0[2][2] - M0[0][0]*M0[1][2]*M0[2][1];
        double detM1 = M1[0][0]*M1[1][1]*M1[2][2] + M1[0][1]*M1[1][2]*M1[2][0] + M1[0][2]*M1[1][0]*M1[2][1] - M1[0][2]*M1[1][1]*M1[2][0] - M1[0][1]*M1[1][0]*M1[2][2] - M1[0][0]*M1[1][2]*M1[2][1];
        double detM2 = M2[0][0]*M2[1][1]*M2[2][2] + M2[0][1]*M2[1][2]*M2[2][0] + M2[0][2]*M2[1][0]*M2[2][1] - M2[0][2]*M2[1][1]*M2[2][0] - M2[0][1]*M2[1][0]*M2[2][2] - M2[0][0]*M2[1][2]*M2[2][1];

        double coeff[3] = {detM0/detM, detM1/detM, detM2/detM};

        max_coeff = -coeff[1]/(2*coeff[2]);
    }

    return max_coeff;
}

void ImageViewer::optimizeDM(QImage image)
{
    dm_metric[dm_idx] = getMetric(image,p_metric);

    if (z_idx <= z_max)
    {
        if((dm_metric[dm_idx] >= dm_metric_max) && (dm_idx > 4))
        {
            dm_idx_max = dm_idx;
            dm_metric_max = dm_metric[dm_idx_max];
        }

        if (dm_idx >= 49)
        {
            double dm_c_max = polyfit();
            std::cerr << z_idx << " " << dm_c_max << " " << dm_metric_max << std::endl;
            moveDM(dm_current_opt,dm_c_max);
            z_idx++;
            dm_idx = 0;
            dm_idx_max = 0;
            for(int i = 0; i < 50; i++)
            {
                dm_c[i] = Z2C[z_idx][97]+(Z2C[z_idx][98]-Z2C[z_idx][97])/50*i;
                dm_metric[i] = 0;
            }
        } else
        {
            std::cerr << z_idx << " " << dm_c[dm_idx] << " " << dm_metric[dm_idx] << std::endl;
            dm_idx++;
            moveDM(dm_data,dm_c[dm_idx]);
        }
    }
    else
    {
        // If we get here, we finished optimization.
        is_dm_optimization = false;
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
