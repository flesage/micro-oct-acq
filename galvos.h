#ifndef GALVOS_H
#define GALVOS_H

#include <QThread>
#include <QString>
#include <QVector>
#include <QMutex>

#include "NIDAQmx.h"
#include "converter.h"

class Galvos : public QThread
{
    Q_OBJECT
public:
    Galvos(QString device, QString ao_x, QString ao_y);
    void setGalvoAxes(QString ao_x, QString ao_y);
    void config();
    void setUnitConverter(Converter converter);
    //void setSynchronizedAITask(a_task);
    void setSawToothRamp(float x0,float y0,float xe,float ye,int nx,int ny,int n_extra,int n_repeat,float line_rate,int aline_repeat);
    void setTriangularRamp(float x0,float y0,float xe,float ye,int nx,int ny,int n_extra,float line_rate);
    void setLineRamp(float x0,float y0,float xe,float ye,int npts,int n_lines,int n_extra,float line_rate);
    void setTrigDelay(float trigdelay);
    void continuousWrite();
    void move(float center_x,float center_y);
    void startTask();
    void stopTask();
    void stopNoClearTask();
    void clearTask();

private:
    void run();

    QString p_device;
    QString p_ao_x;
    QString p_ao_y;
    QString p_camera_clock;
    QString p_camera_clock_pfi;
    QMutex  p_mutex;
    QMutex  p_center_mutex;
    Converter p_converter;
    float p_center_x;
    float p_center_y;
    int p_nx;
    int p_ny;
    int p_n_extra;
    int p_n_repeat;
    float p_line_rate;
    int p_ramp_type;
    int p_daq_freq;
    int p_n_pts_frame;
    TaskHandle p_task_handle;
    TaskHandle p_clock_task_handle;
    int32 p_error;
    bool p_started;
    QVector<double> p_volt_ramp_x;
    QVector<double> p_volt_ramp_y;
    float p_trigdelay;
};

#endif // GALVOS_H
