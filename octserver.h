#ifndef OCTSERVER_H
#define OCTSERVER_H

#include <QDialog>
#include <QLabel>
#include <QTcpServer>
#include <QWidget>
#include <QDir>
#include <QMutex>

class OCTServer : public QDialog
{
    enum RequestMode
    {
        REQUEST_TILE = 0,
        REQUEST_BSCAN_NOSAVE = 1,
        REQUEST_CONFIG = 2,
        REQUEST_UNKNOWN = 3,
    };

    Q_OBJECT

public:
    explicit OCTServer(QWidget *parent = nullptr);
    ~OCTServer();
    void setup_and_request_scan(int x, int y, int z);
    void setup_and_request_scan();

signals:
    void sig_set_request_type(QString);
    void sig_start_scan();
    void sig_change_filename(QString);
    void sig_config_nx(QString);
    void sig_config_ny(QString);
    void sig_config_fov_x(QString);
    void sig_config_fov_y(QString);
    void sig_config_pos(float x , float y, float z);
    void sig_config_zmin(QString);
    void sig_config_zmax(QString);

private slots:
    void slot_startConnection();
    void slot_endConnection();
    void slot_endConnectionAndSendImage(int nx, int ny, int nz, float* image_buffer);
    void slot_parseRequest();

private:
    void initServer();
    QString getHostAddress();

    QLabel *statusLabel = nullptr;
    QTcpServer *tcpServer = nullptr;
    QTcpSocket *clientConnection = nullptr;
    int p_tile_x;
    int p_tile_y;
    int p_tile_z;
    int p_request_type;
    bool p_transfer_requested;
    QMutex p_mutex;
};

#endif // OCTSERVER_H
