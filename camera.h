#ifndef CAMERA_H
#define CAMERA_H

#include <QThread>
#include <QMutex>
#include <QVector>
#include "config.h"
#include "fringeviewer.h"
#include "imageviewer.h"
#include "datasaver.h"
#include "niimaq.h"


class Camera : public QThread
{
    Q_OBJECT
public:
    Camera(int n_lines, float exposure);
    void Open();
    void Close();
    void ConfigureForSingleGrab();
    void Start();
    void Stop();
    void run();
    void SetCameraString(const char* attribute, const char *value);
    void SetCameraNumeric(const char* attribute, double value);
    void setFringeViewer(FringeViewer* ptr);
    void setImageViewer(ImageViewer* ptr);
    void setDataSaver(DataSaver* ptr);
private:
    SESSION_ID sid;
    INTERFACE_ID iid;
    BUFLIST_ID bid;
    Int32 p_error;
    Int8* p_imaq_buffers[NUM_GRAB_BUFFERS];
    int p_n_lines;
    float p_exposure;
    int p_bufsize;
    bool p_started;
    QMutex p_mutex;
    unsigned short* p_current_copied_buffer;
    FringeViewer* fv_ptr;
    ImageViewer* imv_ptr;
    DataSaver* dsaver_ptr;
};

#endif // CAMERA_H
