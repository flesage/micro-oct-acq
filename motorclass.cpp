#include "motorclass.h"
#include <iostream>
#include <QThread>

MotorClass::MotorClass()
{
    is_open = false;
    piezo_is_open = false;
    p_piezo_speed = 0;
    p_piezo_moving=0;
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
//        do
//        {
//            port.write("1MD?\r",5);
//            port.read(buffer,4);
//            QThread::sleep(0.2);
//        } while(!done);

        *done=0;
       // port.write("2OR0\r",5);
//        do
//        {
//            port.write("2MD?\r",5);
//            port.read(buffer,4);
//            QThread::sleep(0.2);
//        } while(!done);

        *done=0;
       // port.write("3OR0\r",5);
//        do
//        {
//            port.write("3MD?\r",5);
//            port.read(buffer,4);
//            QThread::sleep(0.2);
//        } while(!done);
    }

}

void MotorClass::move_dx(float dist)
{
    if(is_open)
    {
        port.flush();
        QString tmp=QString("1PR%1\r").arg(dist,0,'f',2);
        port.write(tmp.toUtf8());
//        char data=0;
//        do
//        {
//            QThread::sleep(0.1);
//            port.write("1MD?\r",5);
//            port.read(&data,1);

//        } while(!data);
    }
}

void MotorClass::move_dy(float dist)
{
    if(is_open)
    {
        port.flush();
        QString tmp=QString("2PR%1\r").arg(dist,0,'f',2);
        port.write(tmp.toUtf8());
/*        char data=0;
        do
        {
            QThread::sleep(0.1);
            port.write("2MD?\r",5);
            port.read(&data,1);
        } while(!data)*/;
    }
}

void MotorClass::move_dz(float dist)
{
    if(is_open)
    {
        port.flush();
        QString tmp=QString("3PR%1\r").arg(dist,0,'f',2);;
        port.write(tmp.toUtf8());
//        char data=0;
//        do
//        {
//            QThread::sleep(0.1);
//            port.write("3MD?\r",5);
//            port.read(&data,1);
//        } while(!data);
    }
}

void MotorClass::move_ax(float dist)
{
    if(is_open)
    {
        port.flush();
        QString tmp=QString("1PA%1\r").arg(dist,0,'f',2);
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
        QString tmp=QString("2PA%1f\r").arg(dist,0,'f',2);
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
        QString tmp=QString("3PA%1\r").arg(dist,0,'f',2);
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
        piezo_port.setPortName("COM7");
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
        piezo_port.flush();
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

        //char buffer[4];

        //        Wait until homing is done to return from this function to
        //        make sure we have accurate positioning
        //        Motor 1
        //char piezodone=0;
        piezo_port.flush();
        piezo_port.write("1JA2\r",5);
        p_piezo_moving=1;
        piezo_port.flush();

        //piezo_port.read(&piezodone,1);
        //std::cout<<piezodone<<std::endl;
        /*do
        {
            port.write("1MD?\r",5);
            piezo_port.read(&piezodone,4);
            QThread::sleep(0.2);
        } while(!piezodone);
        *piezodone=0;*/
    }

}

void MotorClass::getSpeed(int speedval)
{
    p_piezo_speed=speedval;
    std::cout<<"speed updated: "<<p_piezo_speed<<std::endl;
    QByteArray buffer2;
    piezo_port.flush();

     QThread::sleep(0.1);
     piezo_port.write("1TS?\r");

     buffer2 = piezo_port.readAll();
     std::cout<<"done: "<<buffer2.constData()<<std::endl;
     QThread::sleep(0.5);

   piezo_port.flush();
}


void MotorClass::PiezoStartJog()
{
    if(piezo_is_open)
    {
        std::cout<<"...moving piezo"<<std::endl;
        QThread::sleep(0.2);
        piezo_port.flush();
        switch(p_piezo_speed)
        {
        case 0:
            piezo_port.write("1JA0\r",5);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA0\r",5);
            std::cout<<"speed 0 written!"<<std::endl;
            break;
        case 1:
            piezo_port.write("1JA1\r",5);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA1\r",5);
            std::cout<<"speed 1 written!"<<std::endl;
            break;
        case 2:
            piezo_port.write("1JA2\r",5);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA2\r",5);
            std::cout<<"speed 2 written!"<<std::endl;
            break;
        case 3:
            piezo_port.write("1JA3\r",5);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA3\r",5);
            std::cout<<"speed 3 written!"<<std::endl;
            break;
        case 4:
            piezo_port.write("1JA4\r",5);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA4\r",5);
            std::cout<<"speed 4 written!"<<std::endl;
            break;
        case 5:
            piezo_port.write("1JA5\r",5);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA5\r",5);
            std::cout<<"speed 5 written!"<<std::endl;
            break;
        case 6:
            piezo_port.write("1JA6\r",5);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA6\r",5);
            std::cout<<"speed 6 written!"<<std::endl;
            break;
        case 7:
            piezo_port.write("1JA7\r",5);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA7\r",5);
            std::cout<<"speed 7 written!"<<std::endl;
            break;
        case -7:
            piezo_port.write("1JA-7\r",6);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA-7\r",6);
            std::cout<<"speed -7 written!"<<std::endl;
            break;
        case -6:
            piezo_port.write("1JA-6\r",6);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA-6\r",6);
            std::cout<<"speed -6 written!"<<std::endl;
            break;
        case -5:
            piezo_port.write("1JA-5\r",6);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA-5\r",6);
            std::cout<<"speed -5 written!"<<std::endl;
            break;
        case -4:
            piezo_port.write("1JA-4\r",6);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA-4\r",6);
            std::cout<<"speed -4 written!"<<std::endl;
            break;
        case -3:
            piezo_port.write("1JA-3\r",6);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA-3\r",6);
            std::cout<<"speed -3 written!"<<std::endl;
            break;
        case -2:
            piezo_port.write("1JA-2\r",6);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA-2\r",6);
            std::cout<<"speed -2 written!"<<std::endl;
            break;
        case -1:
            piezo_port.write("1JA-1\r",6);
            QThread::sleep(0.5);
            piezo_port.flush();
            piezo_port.write("1JA-1\r",6);
            std::cout<<"speed -1 written!"<<std::endl;
            break;
        default:
            std::cout<<"nothing is happening"<<std::endl;
            break;
        }
        std::cout<<"...writing over"<<std::endl;
        piezo_port.flush();
    }

}

void MotorClass::PiezoStopJog()
{
    if(piezo_is_open)
    {
        std::cout<<"...stopping piezo"<<std::endl;
        piezo_port.flush();

        char buffer=0;

        //        Wait until homing is done to return from this function to
        //        make sure we have accurate positioning
        //        Motor 1
        piezo_port.write("1ST\r",5);
        piezo_port.flush();
        p_piezo_moving=0;
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
