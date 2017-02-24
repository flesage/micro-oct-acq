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

#ifndef ANALOGINPUT_H
#define ANALOGINPUT_H

#include <Qthread>
#include <QMutex>
#include "nidaqmx.h"
#include "float64datasaver.h"

class AnalogInput : public QThread
{
    Q_OBJECT
public:
    AnalogInput(float64 sample_rate);
    void Start();
    void Stop();
    void run();
    void SetDataSaver(Float64DataSaver* data_saver_ptr);
private:
    TaskHandle p_ai_task_handle;
    Float64DataSaver* p_data_saver_ptr;
    bool p_started;
    QMutex p_mutex;
    int32 p_error;
};

#endif // ANALOGINPUT_H
