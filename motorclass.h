#ifndef MOTORCLASS_H
#define MOTORCLASS_H
#include <QtSerialPort>

class MotorClass
{
public:
    MotorClass();
    virtual ~MotorClass();
    void OpenPort();
    void ClosePort();
    void Home();
    void move_dx(float dist);
    void move_dy(float dist);
    void move_dz(float dist);
    void move_ax(float dist);
    void move_ay(float dist);
    void move_az(float dist);

private:
    QSerialPort port;
    bool is_open;
};

#endif // MOTORCLASS_H


