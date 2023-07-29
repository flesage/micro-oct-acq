#include "saver_remote.h"
#include <stdio.h>
#include <QDir>
#include <iostream>
#include "config.h"


Saver_Remote::Saver_Remote(int n_alines,
                           unsigned int n_repeat,
                           int factor,
                           int save_block_size) :
    f_fft(n_repeat, factor),
    p_n_alines(n_alines),
    p_n_repeat(n_repeat),
    p_factor(factor),
    p_save_block_size(save_block_size),
    p_free_spots(2*p_save_block_size),
    p_used_spots(0),
    p_current_pos(0)
{
    p_frame_size = p_n_alines * p_factor * LINE_ARRAY_SIZE;
    p_buffer_size = 2 * p_save_block_size;
    p_fringe_buffer = new unsigned short[p_frame_size * p_buffer_size];
    p_image_buffer = new float[p_frame_size/2];

    p_started = false;
    p_transfer_started = false;

    // Prepare the data reconstruction
    float dimz = 3.5; // FIXME: dummy axial resolution
    float dimx = 3.0; // FIXME: dummy lateral resolution
    f_fft.init(LINE_ARRAY_SIZE, p_n_alines * p_factor, dimz, dimx);

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
}

void Saver_Remote::put(unsigned short* frame)
{
    // Copying the frame in the next available space of the rolling buffer.
    // Will wait until a space is available
    p_free_spots.acquire();
    memcpy(&p_fringe_buffer[(p_current_pos % p_buffer_size)*p_frame_size],
           frame,
           p_frame_size*sizeof(unsigned short));
    p_used_spots.release();
    p_current_pos+=1;
}

void Saver_Remote::run()
{
    unsigned int index = 0;
    while (true)
    {
        // Acquire and reconstruct a block of data
        p_used_spots.acquire();
        f_fft.image_reconstruction(&p_fringe_buffer[(index % p_buffer_size) * p_frame_size], p_image_buffer, 1, LINE_ARRAY_SIZE/2);
        emit sig_dimsAndImage(p_n_alines, p_factor, LINE_ARRAY_SIZE/2, p_image_buffer);
        p_free_spots.release();
        index++;

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

