#include "motorclass.h"
#include <iostream>
MotorClass::MotorClass()
{


    port.setPortName("COM9");
    std::cerr << port.open(QIODevice::ReadWrite) << std::endl;
    qDebug() << port.errorString();

    port.setBaudRate(QSerialPort::Baud9600);
    port.setDataBits(QSerialPort::Data8);
    port.setParity(QSerialPort::NoParity);
    port.setStopBits(QSerialPort::OneStop);
    port.setFlowControl(QSerialPort::NoFlowControl);

    port.flush();
    std::cerr << port.write("1MO\r", 4) << std::endl;
//        self.ser.open()
//        self.ser.flushInput()
//        self.ser.flushOutput()
//        self.ser.write('1MO\r')
//        self.ser.write('2MO\r')
//        self.ser.write('3MO\r')
//        self.home()
}

MotorClass::~MotorClass()
{
    port.close();

}

void MotorClass::Home()
{
/*
  #        Wait until homing is done to return from this function to
  #        make sure we have accurate positioning
          #Motor 1
          print('Homing X Motor')
          done=0
          self.ser.write('1OR0\r')
          #self.ser.write('1PA0\r')
          while not done:
                  self.ser.write('1MD?\r')
                  done = self.ser.read(4);
                  time.sleep(1)

          #self.ser.write('1PA0\r')
          #Motor 2
          #=======================================================================
          print('Homing Y Motor')
          done=0
          self.ser.write('2OR0\r')
          while not done:
                  self.ser.write('2MD?\r')
                  done = self.ser.read(4);
                  time.sleep(1)
          #=======================================================================
          #Motor 3
          #=======================================================================
          print('Homing Z Motor')
          done=0
          self.ser.write('3OR0\r')
          while not done:
                  self.ser.write('3MD?\r')
                  done = self.ser.read(4);
                  time.sleep(1)
          #=======================================================================

  #        Center each motor in the center of their stride

          #Motor 1
          #self.move_dx(-27)
          #self.ser.write('1PA12.5\r')
          #while not done:
          #        self.ser.write('1MD?\r')
          #        done = self.ser.read(4);
          #        time.sleep(1)
          */

}


/*
def move_dx(self,dx):
    #print(dx)
    self.ser.flushInput()
    self.ser.flushOutput()
    self.ser.write('1PR'+str(dx)+'\r')
    data=0
    while not data:
        time.sleep(0.1)
        self.ser.write('1MD?\r')
        data = self.ser.read(1);

def move_dy(self,dy):
    self.ser.flushInput()
    self.ser.flushOutput()
    data=0
    self.ser.write('2PR'+str(dy)+'\r')
    while not data:
        self.ser.write('2MD?\r')
        data = self.ser.read(1);
        time.sleep(0.1)

def move_dz(self,dz):
    self.ser.flushInput()
    self.ser.flushOutput()
    done=0
    self.ser.write('3PR'+str(dz)+'\r')
    while not done:
        self.ser.write('3MD?\r')
        done = self.ser.read(1);
        time.sleep(0.1)

def move_ax(self,x_pos):
    self.ser.flushInput()
    self.ser.flushOutput()
    data=0
    self.ser.write('1PA'+str(x_pos)+'\r')
    while not data:
        self.ser.write('1MD?\r')
        data = self.ser.read();
        time.sleep(0.1)

def move_ay(self,y_pos):
    self.ser.flushInput()
    self.ser.flushOutput()
    done=0
    self.ser.write('2PA'+str(y_pos)+'\r')
    while not done:
        self.ser.write('2MD?\r')
        done = self.ser.read(1000);
        time.sleep(0.1)

def move_az(self,z_pos):
    self.ser.flushInput()
    self.ser.flushOutput()
    self.ser.write('3PA'+str(z_pos)+'\r')
    done=0
    while not done:
        self.ser.write('3MD?\r')
        done = self.ser.read(1000);
        time.sleep(0.1)

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
