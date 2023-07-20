#ifndef CAMERA_H
#define CAMERA_H

#include <QThread>
#include <QMutex>
#include <QVector>
#include "config.h"
#include "fringeviewer.h"
#include "imageviewer.h"
#include "oct3dorthogonalviewer.h"
#include "datasaver.h"
#include "imagedatasaver.h"
#include "niimaq.h"
#include "octserver.h"


class Camera : public QThread
{
    Q_OBJECT
public:
    Camera(int n_lines, float exposure, unsigned int n_frames_per_volume);
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
    void set3dViewer(oct3dOrthogonalViewer* ptr);
    void setDataSaver(DataSaver* ptr);
    void setServerDataSaver(OCTServer* ptr);
    void setImageDataSaver(ImageDataSaver* ptr);

signals:
    void volume_done();

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
    oct3dOrthogonalViewer* im3dv_ptr;
    DataSaver* dsaver_ptr;
    ImageDataSaver* imsaver_ptr;
    OCTServer* server_saver_ptr;
    unsigned int p_n_frames_per_volume;
};

#endif // CAMERA_H
