#ifndef SoftwareCamera_H
#define SoftwareCamera_H

#include <Qthread>
#include "fringeviewer.h"
#include "imageviewer.h"
#include "datasaver.h"

class SoftwareCamera : public QThread
{
    Q_OBJECT
public:
    SoftwareCamera(int n_lines, float exposure);
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
    void setDataSaver(DataSaver* ptr);

private:
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

#endif // SoftwareCamera_H
