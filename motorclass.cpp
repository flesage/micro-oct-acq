#include "motorclass.h"
#include <iostream>
#include <QThread>
#include "config.h"

MotorClass::MotorClass()
{
    is_open = false;
    piezo_is_open = false;
    is_open_rotation = false;
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

/* Rotation Stage Controls */

// TODO: Complete and document the method to open the rotation stage port
void MotorClass::RotationStageOpenPort()
{
    std::cout<<"...opening rotation stage"<<std::endl;
    if(!is_open_rotation)
    {
        port_rotation.setPortName(ROTATION_STAGE_PORT_COM);
        port_rotation.open(QSerialPort::ReadWrite);
        port_rotation.setBaudRate(115200);
        port_rotation.setDataBits(QSerialPort::Data8);
        port_rotation.setParity(QSerialPort::NoParity);
        port_rotation.setStopBits(QSerialPort::OneStop);
        port_rotation.setFlowControl(QSerialPort::HardwareControl); // for RTS/CTS hardware flow control
        //rotation_stage_port.write(); // TODO: write the identify command
        port_rotation.flush();
        is_open_rotation=true;
    }
    std::cout<<"...done!"<<std::endl;
}

// TODO: Complete and document the method to close the rotation stage port
void MotorClass::RotationStageClosePort()
{
    std::cout<<"...closing rotation stage"<<std::endl;
    if(is_open_rotation)
    {
        port_rotation.flush();
        //rotation_stage_port.write("1MF\r", 4);
        //rotation_stage_port.flush();
        port_rotation.close();
        is_open_rotation=false;
    }
    std::cout<<"...done!"<<std::endl;
}

// TODO: Rotation Stage Identify
void MotorClass::RotationStageIdentify()
{
    if(is_open_rotation)
    {
        std::cout<<"...identifyng rotation stage"<<std::endl;
        char command[6];
        command[0] = 0x23; // MGMSG_MOD_IDENTIFY: 23/02
        command[1] = 0x02;
        command[2] = 0x01; // Channel Idents 0x01, 0x02, ...
        command[3] = 0x00;
        command[4] = 0x50; // destination, general USB Port
        command[5] = 0x01; // source, host

        // Write the command
        port_rotation.flush();
        port_rotation.write(command, 6);
        port_rotation.flush();
    } else {
        std::cout << "...rotation stage port is not open"<<std::endl;
    }
}

// TODO: Rotation Stage Home
void MotorClass::RotationStageHome()
{
    if(is_open_rotation)
    {
        std::cout<<"...homing the rotation stage"<<std::endl;
        char command[6];
        command[0] = 0x43; // MGMSG_MOT_MOVE_HOME: 43/04
        command[1] = 0x04;
        command[2] = 0x01; // Channel Idents 0x01, 0x02, ...
        command[3] = 0x00;
        command[4] = 0x50; // destination, general USB Port
        command[5] = 0x01; // source, host

        // Write the command
        port_rotation.flush();
        port_rotation.write(command, 6);
        port_rotation.flush();
    } else {
        std::cout << "...rotation stage port is not open"<<std::endl;
    }
}

// Set the rotation stage jog parameters
// TODO: remove thorlabs rotation code from this class
void MotorClass::RotationStageSetJogParameters(float jogStepSize, float jogMaxVelocity, float jogAcceleration)
{
    std::cout<<"...setting the rotation stage jog parameters"<<std::endl;

    // Preparing the command
    char command[28];
    // Header
    command[0] = 0x16;
    command[1] = 0x04;
    command[2] = 0x16;
    command[3] = 0x00;
    command[4] = 0x50; // destination, general USB port
    command[5] = 0x01; // source, host

    // Channel 
    command[6] = 0x01;
    command[7] = 0x00;

    // Jog Mode (1: continuous jogging, 2: single step jogging.)
    command[8] = 0x02;
    command[9] = 0x00;

    // Jog Step Size in encoder counts (conversion described in section 8), long
    long step_size = jogStepSize * ROTATION_DEG2ENC;
    // Conversion to little endien 4 byte array
    command[10] = step_size & 0xFF;
    command[11] = (step_size >> 8) & 0xFF;
    command[12] = (step_size >> 16) & 0xFF;
    command[13] = (step_size >> 24) & 0xFF;

    // Jog Min Velocity (minimum start velocity in encoder counts/sec)
    command[14] = 0x00;
    command[15] = 0x00;
    command[16] = 0x00;
    command[17] = 0x00;

    // Jog Acceleration
    long acceleration = jogAcceleration * ROTATION_ACC2ENC; // deg/sec^2 converted to enc/sec^2
    command[18] = acceleration & 0xFF;
    command[19] = (acceleration >> 8) & 0xFF;
    command[20] = (acceleration >> 16) & 0xFF;
    command[21] = (acceleration >> 24) & 0xFF;

    // Jog Max Velocity
    long velocity = jogMaxVelocity * ROTATION_VEL2ENC; // deg/sec converted to enc/sec
    command[22] = velocity & 0xFF;
    command[23] = (velocity >> 8) & 0xFF;
    command[24] = (velocity >> 16) & 0xFF;
    command[25] = (velocity >> 24) & 0xFF;

    // Stop Mode (1: abrupt, 2: profiled stop or controlled deceleration)
    command[26] = 0x02;
    command[27] = 0x00;

    // DEBUG
    // for (const int &n : command) {
    //     std::cout << std::hex << n << std::dec << " ";
    // }
    // std::cout << std::endl;

    // Writing the command
    port_rotation.flush();
    port_rotation.write(command, 28);
    port_rotation.flush();
}

// Rotation Stage Jog
// @param direction: if >0 the direction is forward, otherwise it is reverse.
// TODO: test this
void MotorClass::RotationStageJog(int direction) 
{
    if (is_open_rotation)
    {
        std::cout<<"...jogging the rotation stage"<<std::endl;
        char command[6];
        command[0] = 0x6A; // MGMSG_MOT_MOVE_JOG 0x046A
        command[1] = 0x04;
        command[2] = 0x01; // Channel: 0x01
        command[3] = (direction > 0)? 0x01: 0x02; // Direction: 0x01 is forward, 0x02 is reverse.
        command[4] = 0x50; // destination, general USB Port
        command[5] = 0x01; // source, host

        // Write the command
        port_rotation.flush();
        port_rotation.write(command, 6);
        port_rotation.flush();
    } else {
        std::cout << "...rotation stage port is not open"<<std::endl;
    }
}

// TODO: test the rotation stage absolute move method
// MGMSG_MOT_MOVE_ABSOLUTE (0x0453) to configure and start an absolute move
void MotorClass::RotationAbsoluteMove(float position)
{
    // Prepare the command
    std::cout<<"...performing a rotation stage absolute move"<<std::endl;
    char command[12];
    command[0] = 0x53; // MGMSG_ MOT_SET_MOVEABSPARAMS (0x0450)
    command[1] = 0x04;
    command[2] = 0x06;
    command[3] = 0x00;
    command[4] = 0x50; // destination, general USB Port
    command[5] = 0x01; // source, host
    command[6] = 0x01; // Channel as a short int
    command[7] = 0x00;

    // Convert position to little endian long int
    long pos_enc = position * ROTATION_DEG2ENC;
    command[8] = pos_enc & 0xFF;
    command[9] = (pos_enc >> 8) & 0xFF;
    command[10] = (pos_enc >> 16) & 0xFF;
    command[11] = (pos_enc >> 24) & 0xFF;

    // Write the configuration command
    port_rotation.flush();
    port_rotation.write(command, 12);
    port_rotation.flush();
}

// Rotation stage, get jog parameters
void MotorClass::RotationStageGetJogParameters()
{
    // Prepare the command
    std::cout<<"...reading the rotation stage jog parameters"<<std::endl;
    char command[6];
    command[0] = 0x17; // MGMSG_MOT_REQ_JOGPARAMS 0x0417
    command[1] = 0x04;
    command[2] = 0x01; // Channel
    command[3] = 0x00;
    command[4] = 0x50; // destination, general USB Port
    command[5] = 0x01; // source, host

    // Write the command, and read the response
    QByteArray response;
    port_rotation.flush();
    QThread::sleep(0.2);
    port_rotation.write(command, 6);
    response = piezo_port.readAll();
    std::cout<<"done: "<<response.constData()<<std::endl;
    QThread::sleep(0.5);
    port_rotation.flush();
}

// TODO: component to display the current position
// TODO: label to show the active / inactive state
// TODO: method to update the motor status in the UI and in the class
// TODO: add method to update the rotation stage parameters based on UI value.

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
