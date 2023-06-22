/*
 * This file is part of the IOI distribution (https://github.com/xxxx).
 * Copyright (c) 2017 Frederic Lesage.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include "analoginput.h"
#include "daqexception.h"
#include "config.h"

#define DAQmxErrChk(functionCall) if( DAQmxFailed(p_error=(functionCall)) ) throw DAQException(p_error) ; else

AnalogInput::AnalogInput(float64 sample_rate): p_started(false)
{
    DAQmxErrChk(DAQmxCreateTask("AnalogInput",&p_ai_task_handle));
    DAQmxErrChk(DAQmxCreateAIVoltageChan(p_ai_task_handle,AICHANNELS,"",DAQmx_Val_RSE,-10.0,10.0,DAQmx_Val_Volts,NULL));
    DAQmxErrChk(DAQmxCfgSampClkTiming(p_ai_task_handle,NULL,sample_rate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,(uInt64) sample_rate));
    DAQmxErrChk(DAQmxCfgDigEdgeStartTrig(p_ai_task_handle,CAMERA_CLOCK_PFI,DAQmx_Val_Rising));
}

void AnalogInput::Start()
{
    if (!p_started)
    {
        p_started = true;
        DAQmxErrChk(DAQmxStartTask(p_ai_task_handle));
        start();
    }
}

void AnalogInput::Stop()
{
    std::cerr << "AnalogInput::Stop()" << std::endl;
    // Stop reading first as reading calls could block
    p_mutex.lock();
    p_started = false;
    p_mutex.unlock();
    wait();
    DAQmxErrChk(DAQmxStopTask(p_ai_task_handle));
    DAQmxErrChk(DAQmxClearTask(p_ai_task_handle));
}



void AnalogInput::run()
{
    // Read every second and send to saver
    int32 num_samp_per_chan = AIAOSAMPRATE;
    int32 samples_read;
    uInt32 arraySizeInSamps = N_AI_CHANNELS*num_samp_per_chan;
    float64* data = new float64[arraySizeInSamps];

    while(true)
    {
        DAQmxErrChk(DAQmxReadAnalogF64(p_ai_task_handle,num_samp_per_chan,-1,DAQmx_Val_GroupByChannel,data,arraySizeInSamps,&samples_read,NULL));

        if(p_data_saver_ptr)
        {
            p_data_saver_ptr->put((float64*) data);
        }
        // Needs to be fast
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
    delete [] data;
}

void AnalogInput::SetDataSaver(Float64DataSaver* data_saver_ptr)
{
    p_data_saver_ptr = data_saver_ptr;
}
