#include <iostream>
#include "galvos.h"
#include "daqexception.h"
#include "config.h"

#define DAQmxErrChk(functionCall) if( DAQmxFailed(p_error=(functionCall)) ) throw DAQException(p_error) ; else

Galvos::Galvos(QString device, QString ao_x, QString ao_y) : p_device(device), p_ao_x(ao_x), p_ao_y(ao_y)
{

    p_task_handle=0;
    p_started=false;
    p_error=0;
    p_center_x=0.0;
    p_center_y=0.0;
    p_nx=256;
    p_ny=256;
    p_n_extra=0;
    p_n_repeat=1;
    p_line_rate=100;
    p_ramp_type=0;
    p_daq_freq=int(p_line_rate*(p_nx+p_n_extra));
    // This sets a write rate of 2Hz which will not take too much CPU but still stop quickly
    p_n_pts_frame = int((p_nx+p_n_extra)*p_line_rate/2);

    p_camera_clock = CAMERA_CLOCK;
    p_camera_clock_pfi = CAMERA_CLOCK_PFI;
    p_trigdelay = 0.0;
}

void Galvos::setGalvoAxes(QString ao_fast, QString ao_slow)
{
    p_ao_x=ao_fast;
    p_ao_y=ao_slow;
    std::cout<<"axes inverted"<<std::endl;
    //std::cout<<'fast axis: '+ ao_fast.toStdString() + ' ; slow axis: ' + ao_slow.toStdString()<<std::endl;
    //std::cout<<'p_ao_x: '+ p_ao_x.toUtf8().constData() + ' ; p_ao_y: ' + p_ao_y.toUtf8().constData()<<std::endl;
}

void Galvos::setTrigDelay(float trigdelay)
{
    p_trigdelay = trigdelay;
}

void Galvos::config()
{
    try
    {
        QString tmp=p_device+"/"+p_camera_clock;
        DAQmxErrChk(DAQmxCreateTask("ClockTask",&p_clock_task_handle));
        DAQmxErrChk (DAQmxCreateCOPulseChanFreq(p_clock_task_handle,tmp.toUtf8().constData(),"",DAQmx_Val_Hz,DAQmx_Val_Low,0.0,p_daq_freq,0.50));
        DAQmxErrChk (DAQmxCfgImplicitTiming(p_clock_task_handle,DAQmx_Val_ContSamps,1000));

        // Counter generated clock will trigger everyone (camera and galvos). Trig galvos to pfi clock but delay a little bit for camera sync.
        DAQmxErrChk(DAQmxCreateTask("GalvosTask",&p_task_handle));
        tmp =  p_device+"/"+p_ao_x+","+p_device+"/"+p_ao_y;
        DAQmxErrChk(DAQmxCreateAOVoltageChan(p_task_handle, tmp.toUtf8().constData(),"Galvos",-10.0,10.0,DAQmx_Val_Volts,NULL));
        tmp=p_device+"/"+p_camera_clock_pfi;
        DAQmxErrChk(DAQmxCfgSampClkTiming(p_task_handle,tmp.toUtf8().constData(),p_daq_freq,DAQmx_Val_Rising,DAQmx_Val_ContSamps,int(p_n_pts_frame)));
        DAQmxErrChk(DAQmxCfgOutputBuffer(p_task_handle,2*int(p_n_pts_frame)));
        DAQmxErrChk(DAQmxSetWriteRegenMode(p_task_handle,DAQmx_Val_DoNotAllowRegen));
        DAQmxErrChk(DAQmxCfgDigEdgeStartTrig(p_task_handle,tmp.toUtf8().constData(),DAQmx_Val_Rising));


        //DAQmxErrChk(DAQmxCfgSampClkTiming(p_task_handle,NULL,p_daq_freq,DAQmx_Val_Rising,DAQmx_Val_ContSamps,int(p_n_pts_frame)));
        //DAQmxErrChk(DAQmxConnectTerms("/Dev1/ao/SampleClock",tmp.toUtf8().constData(),DAQmx_Val_DoNotInvertPolarity));


        //    if self.ai_task is not None:
        //        self.ai_task.config(int(self.n_pts_frame),self.daq_freq)
        //        self.ai_task.CfgDigEdgeStartTrig (posixpath.join(self.device,"ao/StartTrigger"), DAQmx_Val_Falling)
    }
    catch(DAQException e)
    {
        e.show();
        clearTask();
    }
}

void Galvos::clearTask()
{
    p_mutex.lock();
    if (p_started)
    {
        p_mutex.unlock();
        stopTask();
    }
    else
    {
        p_mutex.unlock();
    }
    // Choice is to stop writing to galvos first, then stop galvo clock
    if( p_task_handle!=0 ) {
        DAQmxStopTask(p_task_handle);
        DAQmxClearTask(p_task_handle);
    }
    if( p_clock_task_handle!=0 ) {
        DAQmxStopTask(p_clock_task_handle);
        DAQmxClearTask(p_clock_task_handle);
    }
}

void Galvos::setUnitConverter(Converter converter)
{
    p_converter = converter;
}

void Galvos::setSawToothRamp(float x0,float y0,float xe,float ye,int nx,int ny,int n_extra,int n_repeat,float line_rate, int aline_repeat)
{
    p_mutex.lock();
    if (p_started)
    {
        p_mutex.unlock();
        stopTask();
    }
    else
    {
        p_mutex.unlock();
    }

    p_ramp_type = 0;
    p_nx = nx;
    p_ny = ny;
    p_n_extra = n_extra;
    p_n_repeat = n_repeat;
    p_line_rate=line_rate;

    // Fast axis is always set to be x
    // n_repeat is always before y
    QVector<double> um_ramp_x((nx+n_extra)*(ny*n_repeat)*aline_repeat);
    QVector<double> um_ramp_y((nx+n_extra)*(ny*n_repeat)*aline_repeat);

    for(int j=0;j<ny;j++)
    {
        if (ny>1)
        {
            for (int k=0;k<n_repeat;k++)
            {
                // ramp
                int ind=(k*(nx+n_extra)+j*(nx+n_extra)*n_repeat)*aline_repeat;
                for(int i=0;i<nx;i++)
                {
                    for(int i_a=0;i_a<aline_repeat;i_a++)
                    {
                        um_ramp_x[ind+aline_repeat*i+i_a]=x0+i*(xe-x0)/(nx-1);
                        um_ramp_y[ind+aline_repeat*i+i_a]=y0+j*(ye-y0)/(ny-1);
                    }
                }
                for(int i=nx;i<nx+n_extra;i++)
                {
                    for(int i_a=0;i_a<aline_repeat;i_a++)
                    {
                        um_ramp_x[ind+aline_repeat*i+i_a]=xe+(i-nx)*(x0-xe)/(n_extra-1);
                        um_ramp_y[ind+aline_repeat*i+i_a]=y0+j*(ye-y0)/(ny-1);
                    }
                }
            }
        }
        else
        {
            for (int k=0;k<n_repeat;k++)
            {
                int ind=(k*(nx+n_extra)+j*(nx+n_extra)*n_repeat)*aline_repeat;
                for(int i=0;i<nx;i++)
                {
                    for(int i_a=0;i_a<aline_repeat;i_a++)
                    {
                        um_ramp_x[ind+aline_repeat*i+i_a]=x0+i*(xe-x0)/(nx-1);
                        um_ramp_y[ind+aline_repeat*i+i_a]=(y0+ye)/2;
                    }
                }
                for(int i=nx;i<nx+n_extra;i++)
                {
                    for(int i_a=0;i_a<aline_repeat;i_a++)
                    {
                        um_ramp_x[ind+aline_repeat*i+i_a]=xe+(i-nx)*(x0-xe)/(n_extra-1);
                        um_ramp_y[ind+aline_repeat*i+i_a]=(y0+ye)/2;
                    }
                }
            }
        }
    }
    p_volt_ramp_x = p_converter.voltX(um_ramp_x);
    p_volt_ramp_y = p_converter.voltY(um_ramp_y);

    p_daq_freq=int(p_line_rate*(p_nx+p_n_extra)*aline_repeat);
    // 2 Hz write rate to card
    p_n_pts_frame = int((p_nx+p_n_extra)*aline_repeat*p_line_rate/2);
    config();
}

void Galvos::setTriangularRamp(float x0,float y0,float xe,float ye,int nx,int ny,int n_extra,float line_rate)
{
    p_mutex.lock();
    if (p_started)
    {
        p_mutex.unlock();
        stopTask();
    }
    else
    {
        p_mutex.unlock();
    }

    p_ramp_type = 1;
    if (ny%2 == 1) ny = ny+1;

    p_nx = nx;
    p_ny = ny;
    p_n_extra = n_extra;
    p_n_repeat = 1;
    p_line_rate=line_rate;

    // Fast axis is always set to be x
    QVector<double> um_ramp_x((nx+n_extra)*ny);
    QVector<double> um_ramp_y((nx+n_extra)*ny);

    for(int j=0;j<ny;j+=2)
    {
        // ramp
        int ind=j*(nx+n_extra);
        int ind2=(j+1)*(nx+n_extra);

        for(int i=0;i<nx;i++)
        {
            um_ramp_x[ind+i]=x0+i*(xe-x0)/(nx-1);
            um_ramp_y[ind+i]=y0+j*(ye-y0)/(ny-1);
            um_ramp_x[ind2+i]=xe+i*(x0-xe)/(nx-1);
            um_ramp_y[ind2+i]=y0+(j+1)*(ye-y0)/(ny-1);
        }

        for(int i=nx;i<nx+n_extra;i++)
        {
            um_ramp_x[ind+i]=xe;
            um_ramp_y[ind+i]=y0+j*(ye-y0)/(ny-1);
            um_ramp_x[ind2+i]=x0;
            um_ramp_y[ind2+i]=y0+(j+1)*(ye-y0)/(ny-1);
        }
    }
    p_volt_ramp_x = p_converter.voltX(um_ramp_x);
    p_volt_ramp_y = p_converter.voltY(um_ramp_y);

    p_daq_freq=int(p_line_rate*(p_nx+p_n_extra));
    // This is to chunk the repeats together for easier analysis
    p_n_pts_frame = int((p_nx+p_n_extra)*p_line_rate/2);
    config();
}

void Galvos::setLineRamp(float x0,float y0,float xe,float ye,int npts,int n_lines,int n_extra,float line_rate)
{
    p_mutex.lock();
    if (p_started)
    {
        p_mutex.unlock();
        stopTask();
    }
    else
    {
        p_mutex.unlock();
    }

    if (n_lines%2 == 1) n_lines = n_lines+1;
    p_ramp_type = 2;
    p_nx = npts;
    p_ny = n_lines;
    p_n_extra = n_extra;
    p_n_repeat = 1;
    p_line_rate=line_rate;

    // Fast axis is always set to be x
    QVector<double> um_ramp_x((p_nx+n_extra)*p_ny);
    QVector<double> um_ramp_y((p_nx+n_extra)*p_ny);

    for(int j=0;j<p_ny;j++)
    {
        // ramp
        int ind=j*(p_nx+n_extra);
        for(int i=0;i<p_nx;i++)
        {
            um_ramp_x[ind+i]=x0+i*(xe-x0)/(p_nx-1);
            um_ramp_y[ind+i]=y0+i*(ye-y0)/(p_nx-1);
        }
        for(int i=p_nx;i<p_nx+n_extra;i++)
        {
            um_ramp_x[ind+i]=xe+(i-p_nx)*(x0-xe)/(n_extra-1);
            um_ramp_y[ind+i]=ye+(i-p_nx)*(y0-ye)/(n_extra-1);
        }
    }
    p_volt_ramp_x = p_converter.voltX(um_ramp_x);
    p_volt_ramp_y = p_converter.voltY(um_ramp_y);

    p_daq_freq=int(p_line_rate*(p_nx+p_n_extra));
    // This is to chunk the repeats together for easier analysis
    p_n_pts_frame = int((p_nx+p_n_extra)*p_line_rate/2);
    config();
}

void Galvos::run()
{
    // Write vector, chosen size to write at decent rythm while not overwhelming CPU
    QVector<double> tmp(2*p_n_pts_frame);
    unsigned int current_idx = 0;
    while (true)
    {
        p_center_mutex.lock();
        for (int i=0;i<p_n_pts_frame;i++)
        {
            tmp[i]=p_volt_ramp_x[current_idx]+p_converter.voltX(p_center_x);
            tmp[i+p_n_pts_frame]=p_volt_ramp_y[current_idx]+p_converter.voltY(p_center_y);
            current_idx = (current_idx+1)%p_volt_ramp_x.size();
        }
        p_center_mutex.unlock();
        DAQmxErrChk(DAQmxWriteAnalogF64(p_task_handle,p_n_pts_frame,TRUE,-1,DAQmx_Val_GroupByChannel,
                                        tmp.data(),
                                        NULL,NULL));

        p_mutex.lock();

        if(!p_started)
        {
            p_mutex.unlock();
            break;
        }
        else
        {
            p_mutex.unlock();
        }
    }
}

void Galvos::move(float center_x,float center_y)
{
    p_center_mutex.lock();
    p_center_x = center_x;
    p_center_y = center_y;
    p_center_mutex.unlock();
}

void Galvos::startTask()
{
    if (!p_started)
    {
        p_started = true;
        start();
        msleep(100);
        DAQmxErrChk(DAQmxStartTask(p_clock_task_handle));
    }
}

void Galvos::stopTask()
{
    p_mutex.lock();
    p_started = false;
    p_mutex.unlock();
    wait();
    clearTask();
}
