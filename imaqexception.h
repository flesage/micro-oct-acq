#ifndef IMAQEXCEPTION_H
#define IMAQEXCEPTION_H

#include <exception>
#include <iostream>
#include <QMessageBox>
#include <QString>
#include <NIDAQmx.h>

class IMAQException: public std::exception
{
public:
    IMAQException(Int32 error);

    virtual const char* what() const throw()
    {
        return "IMAQ Error";
    }
    void show();

private:
    Int32 p_error;

};

inline IMAQException::IMAQException(Int32 error)
{
    p_error = error;
}

inline void IMAQException::show()
{
    if (p_error<0) {
        static Int8 ErrorMessage[256];
        memset(ErrorMessage, 0x00, sizeof(ErrorMessage));

        // converts error code to a message
        imgShowError(p_error, ErrorMessage);
        //QMessageBox msgBox;
        //QString tmp;
        //tmp.sprintf("IMAQ Error: %s\n",ErrorMessage);
        //msgBox.setText(tmp);
        //msgBox.exec();
        std::cerr << ErrorMessage << std::endl;
    }
    return;
}
#endif // IMAQEXCEPTION_H
