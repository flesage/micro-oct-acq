#ifndef DAQEXCEPTION_H
#define DAQEXCEPTION_H

#include <exception>
#include <QMessageBox>
#include <QString>
#include <NIDAQmx.h>

class DAQException: public std::exception
{
public:
    DAQException(int32 error);

    virtual const char* what() const throw()
    {
        char    errBuff[2048]={'\0'};

        if( DAQmxFailed(p_error) )
        {
            DAQmxGetExtendedErrorInfo(errBuff,2048);
        }

        return errBuff;
    }
    void show();

private:
    int32 p_error;

};

inline DAQException::DAQException(int32 error)
{
    p_error = error;
}

inline void DAQException::show()
{
    char    errBuff[2048]={'\0'};

    if( DAQmxFailed(p_error) )
    {
        DAQmxGetExtendedErrorInfo(errBuff,2048);
        QMessageBox msgBox;
        QString tmp;
        tmp.sprintf("DAQmx Error:\n %s\n",errBuff);
        msgBox.setText(tmp);
        msgBox.exec();
    }
    return;
}

#endif // DAQEXCEPTION_H

