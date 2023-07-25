#include <iostream>
#include <QCoreApplication>

#include "softwarecamera.h"

SoftwareCamera::SoftwareCamera(int n_lines, float exposure, unsigned int n_frames_per_volume)
{
    p_n_lines = n_lines;
    p_exposure = exposure;
    p_started=false;
    fv_ptr = 0;
    imv_ptr = 0;
    im3dv_ptr = 0;
    imsaver_ptr=0;
    dsaver_ptr=0;
    server_saver_ptr=0;
    p_current_copied_buffer = 0;
    p_n_frames_per_volume=n_frames_per_volume;

}

SoftwareCamera::~SoftwareCamera()
{
    if (p_current_copied_buffer) free(p_current_copied_buffer);
}

void SoftwareCamera::setFringeViewer(FringeViewer *ptr)
{
    fv_ptr = ptr;
}

void SoftwareCamera::setImageViewer(ImageViewer* ptr)
{
    imv_ptr = ptr;
}

void SoftwareCamera::set3dViewer(oct3dOrthogonalViewer* ptr)
{
    im3dv_ptr = ptr;
}

void SoftwareCamera::setDataSaver(DataSaver* ptr)
{
   dsaver_ptr = ptr;
}


void SoftwareCamera::setImageDataSaver(SaverImage* ptr)
{
    imsaver_ptr = ptr;
}

void SoftwareCamera::setServerDataSaver(OCTServer* ptr)
{
    server_saver_ptr = ptr;
}


void SoftwareCamera::SetCameraString(const char* attribute, const char *value)
{
   return;
}
void SoftwareCamera::SetCameraNumeric(const char* attribute, double value)
{
   return;
}

void SoftwareCamera::Open()
{
    p_bufsize = 2048*p_n_lines;
}

void SoftwareCamera::ConfigureForSingleGrab()
{
    p_current_copied_buffer = (unsigned short*) malloc(p_bufsize * sizeof (unsigned short));
}

void SoftwareCamera::Close()
{
    // free our copy buffer
    if (p_current_copied_buffer != NULL)
    {
        free(p_current_copied_buffer);
        p_current_copied_buffer = NULL;
    }
}

void SoftwareCamera::Start()
{
    if (!p_started)
    {
        p_started = true;
        start();
    }
}

void SoftwareCamera::Stop()
{
    p_mutex.lock();
    p_started = false;
    p_mutex.unlock();
    wait();
}

void SoftwareCamera::run()
{
    unsigned int n_frames_read=0;
    unsigned long frame_time = (unsigned long) ((p_exposure*p_n_lines)/1000);
    int a = 0;
    while(true)
    {
        for(int j=0;j<p_n_lines;j++)
        {
            for(int i=0;i<2048;i++)
            {
                p_current_copied_buffer[j*2048+i]=(unsigned short) (512.0*sin((i+a)/128.0)+1024+a);
            }
            a=rand()%64;
        }
        // Needs to be fast
        if(fv_ptr)
        {
            fv_ptr->put((unsigned short*) p_current_copied_buffer);
        }
        if(imv_ptr)
        {
            imv_ptr->put((unsigned short*) p_current_copied_buffer);
        }
        if (im3dv_ptr)
        {
            im3dv_ptr->put((unsigned short*) p_current_copied_buffer, n_frames_read);
        }
        if(dsaver_ptr)
        {
            dsaver_ptr->put((unsigned short*) p_current_copied_buffer);
        }
        if(imsaver_ptr)
        {
            imsaver_ptr->put((unsigned short*) p_current_copied_buffer);
        }
        if(server_saver_ptr){
            server_saver_ptr->put((unsigned short*) p_current_copied_buffer);
        }

        // Needs to be fast
        n_frames_read++;
        if(n_frames_read % p_n_frames_per_volume==0)
        {
            emit volume_done();
            QCoreApplication::processEvents();
        }

        msleep(frame_time);
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
