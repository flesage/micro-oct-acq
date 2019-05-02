#ifndef MOTORCLASS_H
#define MOTORCLASS_H
#include <QtSerialPort>

class MotorClass
{
public:
    MotorClass();
    virtual ~MotorClass();
    void Home();

private:
    QSerialPort port;
};

#endif // MOTORCLASS_H


