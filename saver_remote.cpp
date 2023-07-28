#include "saver_remote.h"
#include <stdio.h>
#include <QDir>
#include <iostream>
#include "config.h"

// Reconfigure the semaphore for the Thread management?


Saver_Remote::Saver_Remote(int n_alines,
                           unsigned int n_repeat,
                           int factor) :
    f_fft(n_repeat, factor),
    p_n_alines(n_alines),
    p_n_repeat(n_repeat),
    p_factor(factor),
    p_current_pos(0)
{
    p_frame_size = p_n_alines * p_factor * LINE_ARRAY_SIZE;
    p_buffer_size = 2*p_frame_size;
    p_started = false;

    // Prepare the data reconstruction
    float dimz = 3.5; // FIXME: dummy axial resolution
    float dimx = 3.0; // FIXME: dummy lateral resolution
    f_fft.init(LINE_ARRAY_SIZE, p_n_alines * p_factor, dimz, dimx);
    p_fringe_buffer = new unsigned short[p_buffer_size];
    p_image_buffer = new float[p_frame_size/2];
}

Saver_Remote::~Saver_Remote()
{
    delete [] p_fringe_buffer;
    delete [] p_image_buffer;
}

void Saver_Remote::startSaving()
{
    p_mutex.lock();
    p_started = true;
    p_mutex.unlock();
    start();
}

void Saver_Remote::stopSaving()
{
    p_mutex.lock();
    p_started = false;
    p_mutex.unlock();
    wait();

    // TODO: send the data to the server
    std::cerr << "stopping the remote saver" << std::endl;
    emit sig_dimsAndImage(p_n_alines, p_factor, LINE_ARRAY_SIZE/2, p_image_buffer);
}

void Saver_Remote::put(unsigned short* frame)
{
    // Copying the frame in the next available space of the rolling buffer
    memcpy(&p_fringe_buffer[(p_current_pos % p_buffer_size)*p_frame_size], frame, p_frame_size*sizeof(unsigned short));
    p_current_pos++;
}


void Saver_Remote::run()
{
    int z_top = 1;
    int z_bottom = LINE_ARRAY_SIZE/2;

    while (true)
    {
        // Reconstruct a block of data
        f_fft.image_reconstruction(p_fringe_buffer, p_image_buffer, z_top, z_bottom);

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

