#ifndef SaverImage_H
#define SaverImage_H

#include <QThread>
#include <QMutex>
#include <QSemaphore>
#include <QVector>
#include "fringefft.h"

class SaverImage : public QThread
{
    Q_OBJECT
public:
    SaverImage(int n_alines = 100, int save_block_size=512, int p_top_z=0, int p_bottom_z=512, unsigned int n_repeat=1, int factor=1);
    virtual ~SaverImage();
    void setDatasetName(QString name);
    void setDatasetPath(QString path);
    void addInfo(QString new_info);
    void writeInfoFile();
    void startSaving();
    void stopSaving();
    void put(unsigned short* frame);
    void run();    
signals:
    void available(int);
    void filenumber(int);
private:
    FringeFFT f_fft;
    int p_n_alines;
    QString p_dataset_name;
    QString p_path_name;
    QString p_info_txt;
    int p_save_block_size;
    int p_frame_size;
    int p_buffer_size;
    int p_n_alines_in_one_volume;
    int p_top_z;
    int p_bottom_z;
    float* p_truncated_image;
    unsigned short int* p_data_buffer;
    unsigned int p_current_pos;
    QMutex p_mutex;
    QSemaphore p_free_spots;
    QSemaphore p_used_spots;
    bool p_started;
};

#endif // SaverImage_H
