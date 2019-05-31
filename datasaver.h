#ifndef DATASAVER_H
#define DATASAVER_H

#include <QThread>
#include <QMutex>
#include <QSemaphore>
#include <QVector>

class DataSaver : public QThread
{
    Q_OBJECT
public:
    DataSaver(int n_alines = 100, int save_block_size=512);
    virtual ~DataSaver();
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
    QString p_dataset_name;
    QString p_path_name;
    QString p_info_txt;
    int p_save_block_size;
    int p_frame_size;
    int p_buffer_size;
    unsigned short int* p_data_buffer;
    unsigned int p_current_pos;
    QMutex p_mutex;
    QSemaphore p_free_spots;
    QSemaphore p_used_spots;
    bool p_started;
};

#endif // DATASAVER_H
