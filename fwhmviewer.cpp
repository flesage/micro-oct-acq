#include "fwhmviewer.h"
#include <QPainter>
#include <QPaintEvent>
#include <iostream>
FWHMViewer::FWHMViewer(QWidget *parent, unsigned int data_size) : QWidget(parent), p_data_size(data_size)
{
    p_aline = new unsigned char[p_data_size];
    background = QBrush(QColor(255, 255,255));
    linePen = QPen(Qt::black);
    linePen.setWidth(1);
    textPen = QPen(Qt::red);
    textFont.setPixelSize(50);
    resize(512, 400);
}

FWHMViewer::~FWHMViewer()
{
    delete [] p_aline;
}

void FWHMViewer::put(unsigned char* data)
{
    memcpy(p_aline,data,p_data_size*sizeof(unsigned char));
    repaint();
}

void FWHMViewer::paintEvent(QPaintEvent *event)
{
    QSize current_size = size();
    float y_scale = current_size.height()/256.0;
    float x_scale = 1.0*current_size.width()/p_data_size;
    // Compute FWHM
    // Compute max and its pos
    unsigned char max = 0;
    int pos;
    for (unsigned int i=0; i<p_data_size; i+=1)
    {
        if(p_aline[i]>max)
        {
            max=p_aline[i];
            pos=i;
        }
    }
    // Compute FWHM
    int fwhm_min, fwhm_max;
    for (unsigned int i=pos; i<p_data_size; i+=1)
    {
        if(p_aline[i]<max/2)
        {
            fwhm_max=i;
            break;
        }
    }
    for (int i=pos; i>0; i-=1)
    {
        if(p_aline[i]<max/2)
        {
            fwhm_min=i;
            break;
        }
    }
    int fwhm = fwhm_max-fwhm_min;
    QPainter painter;
    painter.begin(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(linePen);
    painter.fillRect(event->rect(), background);
    for (unsigned int i=0; i<p_data_size; i+=1)
    {
        painter.drawLine((int)(i*x_scale),(int) ((256-p_aline[i])*y_scale),(int)((i+1)*x_scale),(int) ((256-p_aline[i+1])*y_scale));
    }
    QString str;
    str.sprintf("%s %d", "FWHM: ", fwhm);
    painter.drawText(QPoint(10,10),str);
    painter.end();
}
