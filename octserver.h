#ifndef OCTSERVER_H
#define OCTSERVER_H

#include <QDialog>
#include <QLabel>
#include <QTcpServer>
#include <QWidget>
#include <QDir>
#include <QMutex>
#include "FringeFFT.h"




class OCTServer : public QDialog
{
    enum RequestMode
    {
        REQUEST_TILE = 0,
        REQUEST_BSCAN_NOSAVE = 1,
    };

    Q_OBJECT

public:
    explicit OCTServer(QWidget *parent = nullptr, int nx=400, int n_extra=40, unsigned int n_alines=64);
    ~OCTServer();
    void put(unsigned short* fringe);
    void setup_and_request_scan(int x, int y, int z);

signals:
    void sig_set_request_type(QString);
    void sig_start_scan();
    void sig_change_filename(QString);

private slots:
    void slot_startConnection();
    void slot_endConnection();
    void slot_parseRequest();

private:
    void initServer();
    QString getHostAddress();

    QLabel *statusLabel = nullptr;
    QTcpServer *tcpServer = nullptr;
    QTcpSocket *clientConnection = nullptr;
    FringeFFT f_fft;
    int p_tile_x;
    int p_tile_y;
    int p_tile_z;
    int p_request_type;
    unsigned short int* p_fringe_buffer;
    float* p_image_buffer;
    QMutex p_mutex;
    unsigned int p_n_alines;
    int p_nx;
    int p_n_extra;

};

#endif // OCTSERVER_H
