#include "camera.h"
#include "imaqexception.h"
#include <iostream>
#include <QCoreApplication>

USER_FUNC  niimaquDisable32bitPhysMemLimitEnforcement(SESSION_ID boardid);

#define errChk(fCall) if (p_error = (fCall), p_error < 0) {throw IMAQException(p_error);} else


Camera::Camera(int n_lines, float exposure, unsigned int n_frames_per_volume)
{
    p_n_lines = n_lines;
    p_exposure = exposure;
    p_started=false;
    fv_ptr = 0;
    imv_ptr = 0;
    dsaver_ptr=0;
    p_n_frames_per_volume=n_frames_per_volume;

}

void Camera::setFringeViewer(FringeViewer *ptr)
{
    fv_ptr = ptr;
}

void Camera::setImageViewer(ImageViewer* ptr)
{
    imv_ptr = ptr;
}

void Camera::setDataSaver(DataSaver* ptr)
{
    dsaver_ptr = ptr;
}

void Camera::SetCameraString(const char* attribute, const char* value)
{
    char tmp[256];
    Int32 ret = imgSetCameraAttributeString(sid,(Int8*) attribute,(Int8*) value);
    if (ret != 0)
    {
        imgShowError(ret,tmp);
        std::cerr << tmp << std::endl;
    }
}

void Camera::SetCameraNumeric(const char* attribute, double value)
{
    char tmp[256];
    Int32 ret = imgSetCameraAttributeNumeric(sid,(Int8*) attribute,(double) value);
    if (ret != 0)
    {
        imgShowError(ret,tmp);
        std::cerr << tmp << std::endl;
    }
}

void Camera::Open()
{
    // OPEN A COMMUNICATION CHANNEL WITH THE CAMERA
    // our camera has been given its proper name in Measurement & Automation Explorer (MAX)
    Int32 width;
    Int32 height;
    Int32 bytes_per_pixel;

    errChk(imgInterfaceOpen ("img0", &iid));
    errChk(imgSessionOpen (iid, &sid));
    errChk(imgSetAttribute2(sid, IMG_ATTR_ACQWINDOW_HEIGHT, p_n_lines));
    errChk(imgSessionLineTrigSource2(sid,IMG_SIGNAL_EXTERNAL,IMG_EXT_TRIG0,IMG_TRIG_POLAR_ACTIVEL,0));
    //errChk(imgSetCameraAttributeNumeric(self.sid, 'Exposure Time', e_time));
    imgGetAttribute(sid, IMG_ATTR_ROI_WIDTH, &width);
    imgGetAttribute(sid, IMG_ATTR_ROI_HEIGHT, &height);
    imgGetAttribute(sid, IMG_ATTR_BYTESPERPIXEL, &bytes_per_pixel);
    p_bufsize = width*height*bytes_per_pixel;
    niimaquDisable32bitPhysMemLimitEnforcement(sid);
}

void Camera::ConfigureForSingleGrab()
{
    int buf_cmd;

    p_current_copied_buffer = (unsigned short*) malloc(p_bufsize * sizeof (Int8));
    errChk(imgCreateBufList(NUM_GRAB_BUFFERS, &bid));

    /* the following configuration assigns the following to buffer list
       element i:
            1) buffer pointer that will contain image
            2) size of the buffer for buffer element i
            3) command to loop when this element is reached
     */
    for (int i = 0; i < NUM_GRAB_BUFFERS; i++)
   {
       errChk(imgCreateBuffer(sid, FALSE, p_bufsize, (void**) &p_imaq_buffers[i]));
       errChk(imgSetBufferElement2(bid, i, IMG_BUFF_ADDRESS, p_imaq_buffers[i]));
       errChk(imgSetBufferElement2(bid, i, IMG_BUFF_SIZE, p_bufsize));
       buf_cmd = (i == (NUM_GRAB_BUFFERS - 1)) ? IMG_CMD_LOOP : IMG_CMD_NEXT;
       errChk(imgSetBufferElement2(bid, i, IMG_BUFF_COMMAND, buf_cmd));
   }

    // lock down the buffers contained in the buffer list
    errChk(imgMemLock(bid));
    // configure the session to use this buffer list
    errChk(imgSessionConfigure(sid, bid));
}

void Camera::Close()
{
    // stop the acquisition
    imgSessionAbort(sid, NULL);

    // unlock the buffers in the buffer list
    if (bid != 0)
        imgMemUnlock(bid);

    // dispose of the buffers
    for (int i = 0; i < NUM_GRAB_BUFFERS; i++)
        if (p_imaq_buffers[i] != NULL)
            imgDisposeBuffer(p_imaq_buffers[i]);

    // close this buffer list
    if (bid != 0)
        imgDisposeBufList(bid, FALSE);

    // free our copy buffer
    if (p_current_copied_buffer != NULL)
        free(p_current_copied_buffer);

    // Close the interface and the session
    if(sid != 0)
        imgClose (sid, TRUE);
    if(iid != 0)
        imgClose (iid, TRUE);
}

void Camera::Start()
{
    if (!p_started)
    {
        p_started = true;
#ifndef SIMULATION
        // start the acquisition, asynchronous
        errChk(imgSessionAcquire(sid, TRUE, NULL));
#endif
        start(QThread::TimeCriticalPriority);
    }
}

void Camera::Stop()
{
    p_mutex.lock();
    p_started = false;
    p_mutex.unlock();
    wait();
}

void Camera::run()
{
    unsigned int n_frames_read=0;
#ifndef SIMULATION

    static int currBufNum = 0;
    static uInt32 actualCopiedBuffer = 0;
#else
    unsigned long frame_time = (unsigned long) ((p_exposure*p_n_lines)/1000);
    int a = 0;
#endif
    while(true)
    {
#ifndef SIMULATION
        try
        {
        //Get the frame after the next Vertical Blank
        errChk(imgGetAttribute (sid, IMG_ATTR_LAST_VALID_FRAME, &currBufNum));
        currBufNum++;

        // Copy the last valid buffer
        errChk(imgSessionCopyBufferByNumber(sid, currBufNum, p_current_copied_buffer, IMG_OVERWRITE_GET_NEWEST, &actualCopiedBuffer, NULL));
        }
        catch(IMAQException& e)
        {
            e.show();
        }

#else
        for(int j=0;j<p_n_lines;j++)
        {
            for(int i=0;i<2048;i++)
            {
                p_current_copied_buffer[j*2048+i]=(unsigned short) (512.0*sin((i+a)/128.0)+1024+a);
            }
            a=rand()%64;
        }
        msleep(frame_time);
#endif
        // Send data to consumers

        // Needs to be fast
        if(fv_ptr)
        {
            fv_ptr->put((unsigned short*) p_current_copied_buffer);
        }
        if(imv_ptr)
        {
            imv_ptr->put((unsigned short*) p_current_copied_buffer);
        }
        if(dsaver_ptr)
        {
            dsaver_ptr->put((unsigned short*) p_current_copied_buffer);
        }
        // Needs to be fast
        p_mutex.lock();
        n_frames_read++;
        if(n_frames_read % p_n_frames_per_volume==0)
        {
            emit volume_done();
            QCoreApplication::processEvents();
        }

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
