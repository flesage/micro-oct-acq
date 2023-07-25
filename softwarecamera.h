#ifndef SoftwareCamera_H
#define SoftwareCamera_H

#include <Qthread>
#include "fringeviewer.h"
#include "imageviewer.h"
#include "oct3dorthogonalviewer.h"
#include "datasaver.h"
#include "saver_image.h"
#include "octserver.h"

class SoftwareCamera : public QThread
{
    Q_OBJECT
public:
    SoftwareCamera(int n_lines, float exposure, unsigned int n_frames_per_volume);
    virtual ~SoftwareCamera();
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
    void setImageDataSaver(SaverImage* ptr);
signals:
    void volume_done();
private:
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
    SaverImage* imsaver_ptr;
    OCTServer* server_saver_ptr;
    unsigned int p_n_frames_per_volume;
};

#endif // SoftwareCamera_H
