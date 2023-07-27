#include "saver_remote.h"
#include <stdio.h>
#include <QDir>
#include <iostream>
#include "config.h"


Saver_Remote::Saver_Remote(int n_alines,
                           int save_block_size,
                           unsigned int n_repeat,
                           int factor) :
    p_n_alines(n_alines), p_save_block_size(save_block_size),
    p_free_spots(2*p_save_block_size), p_used_spots(0), p_current_pos(0),
    f_fft(n_repeat,factor)

{
    p_frame_size = n_alines*LINE_ARRAY_SIZE;
    p_buffer_size = 2*p_save_block_size;
    p_fringe_buffer = new unsigned short[p_frame_size*p_buffer_size];
    p_started = false;
    f_fft.init(LINE_ARRAY_SIZE, p_n_alines, 3.5, 3.5);     //TODO Move dimz dimx to doppler call
    p_image_buffer = new float[(LINE_ARRAY_SIZE/2)*p_n_alines];
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
    p_free_spots.acquire();

    // Copying the frame in the next available space of the rolling buffer
    memcpy(&p_fringe_buffer[(p_current_pos % p_buffer_size)*p_frame_size], frame, p_frame_size*sizeof(unsigned short));

    p_used_spots.release();
    p_current_pos++;

    emit available(p_free_spots.available());
}

void Saver_Remote::run()
{
    unsigned int index = 0;
    while (true)
    {
        // Reconstruct a block of data
        p_used_spots.acquire();
        f_fft.image_reconstruction(&p_fringe_buffer[(index % p_buffer_size)*p_frame_size], p_image_buffer, 0, LINE_ARRAY_SIZE/2);

        // TODO: Add data to the bytearray
        p_free_spots.release();
        index++;

        p_mutex.lock();
        if(!p_started)
        {
            p_mutex.unlock();
            if(p_used_spots.available() == 0) break;
        }
        else
        {
            p_mutex.unlock();
        }
    }
    //if (fp) fclose(fp);
}

