#ifndef FLOAT64DATASAVER_H
#define FLOAT64DATASAVER_H

#include <stdio.h>
#include <iostream>
#include <QThread>
#include <QMutex>
#include <QSemaphore>
#include <QVector>
#include <QDir>
#include "NIDAQmx.h"

// Generic templated data saver with threads. Data is put into the saver in frames of size
// size_x*size_y. Each file contains save_block_size frames.
class Float64DataSaver : public QThread
{
    Q_OBJECT
public:
    Float64DataSaver(int size_x, int size_y, int save_block_size, const char* prefix);
    virtual ~Float64DataSaver();
    void setDatasetName(QString name);
    void setDatasetPath(QString path);
    void addInfo(QString new_info);
    void writeInfoFile();
    void startSaving();
    void stopSaving();
    void put(float64* frame);
    void run();
signals:
    void available(int);
    void filenumber(int);
private:
    QString p_dataset_name;
    QString p_path_name;
    QString p_info_txt;
    QString p_file_prefix;
    int p_save_block_size;
    int p_frame_size;
    int p_size_x;
    int p_size_y;
    int p_buffer_size;
    float64* p_data_buffer;
    unsigned int p_current_pos;
    QMutex p_mutex;
    QSemaphore p_free_spots;
    QSemaphore p_used_spots;
    bool p_started;
};

inline Float64DataSaver::Float64DataSaver(int size_x, int size_y, int save_block_size, const char* prefix) :
    p_save_block_size(save_block_size), p_free_spots(2*p_save_block_size),
    p_used_spots(0), p_current_pos(0), p_file_prefix(prefix)
{
    p_frame_size = size_x*size_y;
    p_size_x = size_x;
    p_size_y = size_y;
    p_buffer_size = 2*p_save_block_size;

    p_data_buffer = new float64[p_frame_size*p_buffer_size];

    p_started = false;
    p_dataset_name = "dummy";
    p_path_name = QDir::homePath();
    p_info_txt="Scan info\n";
}

inline Float64DataSaver::~Float64DataSaver()
{
    delete [] p_data_buffer;
}

inline void Float64DataSaver::addInfo(QString new_info)
{
    p_info_txt += new_info;
}

inline void Float64DataSaver::setDatasetName(QString name)
{
    p_dataset_name = name;
}
inline void Float64DataSaver::setDatasetPath(QString path)
{
    p_path_name = path;
}

inline void Float64DataSaver::startSaving()
{
    p_started = true;
    start();
}

inline void Float64DataSaver::stopSaving()
{
    p_mutex.lock();
    p_started = false;
    p_mutex.unlock();
    wait();
}

inline void Float64DataSaver::put(float64* frame)
{
    p_free_spots.acquire();
    memcpy(&p_data_buffer[(p_current_pos % p_buffer_size)*p_frame_size],frame,p_frame_size*sizeof(float64));
    p_used_spots.release();
    p_current_pos+=1;

    emit available(p_free_spots.available());
}

inline void Float64DataSaver::writeInfoFile()
{
    QDir parent_dir = QDir::cleanPath(p_path_name);
    parent_dir.mkdir(p_dataset_name);
    parent_dir.setPath(QDir::cleanPath(p_path_name + QDir::separator() + p_dataset_name +QDir::separator()));

    QString tmp = "info.txt";
    tmp=parent_dir.absolutePath()+QDir::separator()+tmp;
    FILE* fp = fopen(tmp.toUtf8().constData(), "w");
    fprintf(fp,"%s\n",p_info_txt.toUtf8().constData());
    fclose(fp);
}

inline void Float64DataSaver::run()
{
    QDir parent_dir = QDir::cleanPath(p_path_name);
    parent_dir.mkdir(p_dataset_name);
    parent_dir.setPath(QDir::cleanPath(p_path_name + QDir::separator() + p_dataset_name));
    FILE* fp = 0;
    QString tmp;

    unsigned int file_num = 0;
    unsigned int index = 0;
    while (true)
    {
        if(index % p_save_block_size==0)
        {
            // Change file when we have a chunk
            if (fp) fclose(fp);
            tmp=QString("%1_%2.bin").arg(p_file_prefix.toUtf8()).arg(file_num,5);
            tmp=parent_dir.absolutePath()+ QDir::separator()+tmp;
            fp = fopen(tmp.toUtf8().constData(), "wb");
            // Write header
            int version=1;
            fwrite(&version, sizeof(int), 1, fp);
            fwrite(&p_size_x, sizeof(int), 1, fp);
            fwrite(&p_size_y, sizeof(int), 1, fp);
            fwrite(&p_save_block_size, sizeof(int), 1, fp);
            emit filenumber(file_num);
            file_num++;

        }
        // Acquire a block of data
        p_used_spots.acquire();
        fwrite(&p_data_buffer[(index % p_buffer_size)*p_frame_size], sizeof(float64), p_frame_size, fp);
        fflush(fp);
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
    if (fp) fclose(fp);
}

#endif // FLOAT64DATASAVER_H
