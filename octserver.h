#ifndef OCTSERVER_H
#define OCTSERVER_H

#include <QDialog>
#include <QLabel>
#include <QTcpServer>
#include <QWidget>
#include <QDir>


class OCTServer : public QDialog
{
    Q_OBJECT

public:
    explicit OCTServer(QWidget *parent = nullptr);

signals:
    void sig_start_acquisition(int, int, int);
    void sig_end_acquisition();

private slots:
    void slot_startConnection();
    void slot_endConnection();
    void slot_readTilePosition();
    void slot_performScan(int x, int y, int z);

private:
    void initServer();
    QString getHostAddress();

    QLabel *statusLabel = nullptr;
    QTcpServer *tcpServer = nullptr;
    QTcpSocket *clientConnection = nullptr;
    int p_tile_x;
    int p_tile_y;
    int p_tile_z;
};

#endif // OCTSERVER_H
