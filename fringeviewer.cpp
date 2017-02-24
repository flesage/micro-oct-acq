#include "fringeviewer.h"
#include <QPainter>
#include <QPaintEvent>
#include <iostream>

FringeViewer::FringeViewer(QWidget *parent, int n_alines) :
    QWidget(parent)
{
    background = QBrush(QColor(255, 255,255));
    maxPen = QPen(Qt::red);
    maxPen.setWidth(1);

    linePen = QPen(Qt::black);
    linePen.setWidth(1);
    textPen = QPen(Qt::white);
    textFont.setPixelSize(50);
    p_nalines=n_alines;
    p_data_buffer=new unsigned short[p_nalines*LINE_ARRAY_SIZE];
    resize(512, 400);
}

FringeViewer::~FringeViewer()
{
    delete [] p_data_buffer;
}

void FringeViewer::Close()
{
    close();
}

void FringeViewer::put(unsigned short* data)
{
    if(p_mutex.tryLock())
    {
        memcpy(p_data_buffer,data,p_nalines*LINE_ARRAY_SIZE*sizeof(unsigned short));
        p_mutex.unlock();
    }
}


void FringeViewer::updateView()
{
    float avg_fringe[LINE_ARRAY_SIZE];

    // Critical section
    p_mutex.lock();
    for (int i=0; i<LINE_ARRAY_SIZE; i++)
    {
       avg_fringe[i]=p_data_buffer[i];
       p_max_fringe[i]=p_data_buffer[i];
    }
    for (int j=1 ; j<p_nalines; j++)
    {
        for (int i=0; i<LINE_ARRAY_SIZE; i++)
        {
            avg_fringe[i]+=p_data_buffer[j*LINE_ARRAY_SIZE+i];
            if(p_data_buffer[j*LINE_ARRAY_SIZE+i]>p_max_fringe[i]) p_max_fringe[i] =p_data_buffer[j*LINE_ARRAY_SIZE+i];
        }
    }
    p_mutex.unlock();

    for (int i=0; i<LINE_ARRAY_SIZE; i++)
    {
       p_fringe[i]=(unsigned short) (avg_fringe[i]/p_nalines);
    }
    repaint();
}

void FringeViewer::paintEvent(QPaintEvent *event)
{
    QSize current_size = size();
    float scale = current_size.height()/4096.0;
    int x_step = LINE_ARRAY_SIZE/current_size.width();
    QPainter painter;
    painter.begin(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(linePen);
    painter.fillRect(event->rect(), background);
    int index = 0;
    for (int i=0; i<LINE_ARRAY_SIZE-1-x_step; i+=x_step)
    {
        painter.drawLine(index,(int) ((4096-p_fringe[i])*scale),index+1,(int) ((4096-p_fringe[i+x_step])*scale));
        index+=1;
    }
    painter.setPen(maxPen);

    index = 0;
    for (int i=0; i<LINE_ARRAY_SIZE-1-x_step; i+=x_step)
    {
        painter.drawLine(index,(int) ((4096-p_max_fringe[i])*scale),index+1,(int) ((4096-p_max_fringe[i+x_step])*scale));
        index+=1;
    }
    painter.end();
}
