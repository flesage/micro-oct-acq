#include "motorclass.h"
#include <iostream>
#include <QThread>

MotorClass::MotorClass()
{
    is_open = false;
    piezo_is_open = false;
}

MotorClass::~MotorClass()
{
    ClosePort();
}

void MotorClass::OpenPort()
{
    if(!is_open)
    {
        port.setPortName("COM9");
        port.open(QSerialPort::ReadWrite);
        port.setBaudRate(921600);
        port.setDataBits(QSerialPort::Data8);
        port.setParity(QSerialPort::NoParity);
        port.setStopBits(QSerialPort::OneStop);
        port.setFlowControl(QSerialPort::HardwareControl);
        port.flush();
        port.write("1MO\r", 4);
        port.write("2MO\r", 4);
        port.write("3MO\r", 4);
        is_open=true;
    }
}

void MotorClass::ClosePort()
{
    if(is_open)
    {
        port.write("1MF\r", 4);
        port.write("2MF\r", 4);
        port.write("3MF\r", 4);
        port.flush();
        port.close();
        is_open=false;
    }
    if(piezo_is_open)
    {
        piezo_port.write("1MF\r", 4);
        piezo_port.flush();
        piezo_port.close();
        piezo_is_open=false;
    }
}

void MotorClass::Home()
{
    if(is_open)
    {
        char buffer[4];

        //        Wait until homing is done to return from this function to
        //        make sure we have accurate positioning
        //        Motor 1
        int* done = (int*) &buffer;
        port.write("1OR0\r",5);
        do
        {
            port.write("1MD?\r",5);
            port.read(buffer,4);
            QThread::sleep(0.2);
        } while(!done);

        *done=0;
        port.write("2OR0\r",5);
        do
        {
            port.write("2MD?\r",5);
            port.read(buffer,4);
            QThread::sleep(0.2);
        } while(!done);

        *done=0;
       port.write("3OR0\r",5);
        do
        {
            port.write("3MD?\r",5);
            port.read(buffer,4);
            QThread::sleep(0.2);
        } while(!done);
    }

}

void MotorClass::move_dx(float dist)
{
    if(is_open)
    {
        port.flush();
        QString tmp;
        tmp.sprintf("1PR%.2f\r",dist);
        port.write(tmp.toUtf8());
        char data=0;
        do
        {
            QThread::sleep(0.1);
            port.write("1MD?\r",5);
            port.read(&data,1);
        } while(!data);
    }
}

void MotorClass::move_dy(float dist)
{
    if(is_open)
    {
        port.flush();
        QString tmp;
        tmp.sprintf("2PR%.2f\r",dist);
        port.write(tmp.toUtf8());
        char data=0;
        do
        {
            QThread::sleep(0.1);
            port.write("2MD?\r",5);
            port.read(&data,1);
        } while(!data);
    }
}

void MotorClass::move_dz(float dist)
{
    if(is_open)
    {
        port.flush();
        QString tmp;
        tmp.sprintf("3PR%.2f\r",dist);
        port.write(tmp.toUtf8());
        char data=0;
        do
        {
            QThread::sleep(0.1);
            port.write("3MD?\r",5);
            port.read(&data,1);
        } while(!data);
    }
}

void MotorClass::move_ax(float dist)
{
    if(is_open)
    {
        port.flush();
        QString tmp;
        tmp.sprintf("1PA%.2f\r",dist);
        port.write(tmp.toUtf8());
        char data=0;
        do
        {
            QThread::sleep(0.1);
            port.write("1MD?\r",5);
            port.read(&data,1);
        } while(!data);
    }
}

void MotorClass::move_ay(float dist)
{
    if(is_open)
    {
        port.flush();
        QString tmp;
        tmp.sprintf("2PA%.2f\r",dist);
        port.write(tmp.toUtf8());
        char data=0;
        do
        {
            QThread::sleep(0.1);
            port.write("2MD?\r",5);
            port.read(&data,1);
        } while(!data);
    }
}

void MotorClass::move_az(float dist)
{
    if(is_open)
    {
        port.flush();
        QString tmp;
        tmp.sprintf("3PA%.2f\r",dist);
        port.write(tmp.toUtf8());
        char data=0;
        do
        {
            QThread::sleep(0.1);
            port.write("3MD?\r",5);
            port.read(&data,1);
        } while(!data);
    }
}

void MotorClass::PiezoOpenPort()
{
    std::cout<<"...opening piezo"<<std::endl;
    if(!piezo_is_open)
    {
        piezo_port.setPortName("COM5");
        piezo_port.open(QSerialPort::ReadWrite);
        piezo_port.setBaudRate(19200);
        piezo_port.setDataBits(QSerialPort::Data8);
        piezo_port.setParity(QSerialPort::NoParity);
        piezo_port.setStopBits(QSerialPort::OneStop);
        piezo_port.setFlowControl(QSerialPort::SoftwareControl);

        piezo_port.flush();
        piezo_port.write("1MO\r", 4);
        piezo_is_open=true;
    }
    std::cout<<"...done!"<<std::endl;
}

void MotorClass::PiezoClosePort()
{
    std::cout<<"...closing piezo"<<std::endl;
    if(piezo_is_open)
    {
        piezo_port.write("1MF\r", 4);
        piezo_port.flush();
        piezo_port.close();
        piezo_is_open=false;
    }
    std::cout<<"...done!"<<std::endl;
}

void MotorClass::PiezoHome()
{
    if(piezo_is_open)
    {
        std::cout<<"...homing piezo"<<std::endl;

        char buffer[4];

        //        Wait until homing is done to return from this function to
        //        make sure we have accurate positioning
        //        Motor 1
        int* piezodone = (int*) &buffer;
        piezo_port.write("1JA1\r",5);
        //do
        //{
         //   port.write("1MD?\r",5);
          //  piezodone=port.read(buffer,4);
           // QThread::sleep(0.2);
        //} while(!piezodone);

        //*piezodone=0;
    }

}

//void MotorClass::PiezoHome()
//{
//    if(piezo_is_open)
//    {
//        char buffer[4];

//        //        Wait until homing is done to return from this function to
//        //        make sure we have accurate positioning
//        //        Motor 1
//        int* done = (int*) &buffer;
//        piezo_port.write("1OR\r",5);
//        do
//        {
//            piezo_port.write("1MD?\r",5);
//            done=piezo_port.read(buffer,4);
//            QThread::sleep(0.2);
//        } while(done==81);

//    }

//}

/*



def get_pos_axial(self):
    self.ser.flushInput()
    self.ser.flushOutput()
    print 'get_pos: in function'
    data=0
    while not data:
        self.ser.write('3TP\r')
        data = self.ser.read(1000)
    time.sleep(0.1)
    print 'get_pos: done'
    return data

def get_pos(self,axisVal):
    self.ser.flushInput()
    self.ser.flushOutput()
    print 'get_pos: in function'
    data=0
    while not data:
        self.ser.write(str(axisVal)+'TP\r')
        data = self.ser.read(1000)
    time.sleep(0.1)
    print 'get_pos: done'
    return data

        #self.ser.write('2TP')
#             y = self.ser.read()
#             self.ser.write('3TP')
#             z = self.ser.read()
def getouttheway(self):
    self.ser.write('1PA' + str(13)+'\r')

def delete(self):
    self.ser.write('1MF\r')
#             self.ser.write('2MF')
#             self.ser.write('3MF')

    self.ser.close()
    self.ser.delete()
*/
