#ifndef PIEZOSTAGE_H
#define PIEZOSTAGE_H
#include <QSerialPort>

class piezoStage
{
public:
    piezoStage();
    virtual ~piezoStage();
    void OpenPort();
    void ClosePort();
    void Home();
private:
    QSerialPort port;
    bool is_open;
};

#endif // PIEZOSTAGE_H
