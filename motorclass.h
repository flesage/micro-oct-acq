#ifndef MOTORCLASS_H
#define MOTORCLASS_H
#include <QSerialPort>

class MotorClass
{
public:
    MotorClass();
    virtual ~MotorClass();
    void OpenPort();
    void ClosePort();
    void Home();
    void PiezoOpenPort();
    void PiezoClosePort();
    void PiezoHome();
    void move_dx(float dist);
    void move_dy(float dist);
    void move_dz(float dist);
    void move_ax(float dist);
    void move_ay(float dist);
    void move_az(float dist);

private:
    QSerialPort port;
    QSerialPort piezo_port;
    bool is_open;
    bool piezo_is_open;

};

#endif // MOTORCLASS_H


