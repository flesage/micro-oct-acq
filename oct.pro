#-------------------------------------------------
#
# Project created by QtCreator 2016-03-31T18:23:47
#
#-------------------------------------------------

QT       += core gui serialport

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = oct
TEMPLATE = app
RC_FILE = oct.rc

DEFINES += NOMINMAX

# Add the necessary libraries
INCLUDEPATH += "C:\Program Files (x86)\National Instruments\NI-DAQ\DAQmx ANSI C Dev\include"
INCLUDEPATH += "C:\Program Files (x86)\National Instruments\NI-IMAQ\Include"
INCLUDEPATH += "C:\Program Files\ArrayFire\v3\include"

LIBS += -L"C:\Program Files\ArrayFire\v3\lib" -lafopencl
LIBS += -L"C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\lib64\msvc" -lNIDAQmx
LIBS += -L"C:\Program Files (x86)\National Instruments\NI-IMAQ\Lib\MSVC" -limaq

SOURCES += main.cpp \
    galvocontroller.cpp \
    galvos.cpp \
    converter.cpp \
    camera.cpp \
    datasaver.cpp \
    fringeviewer.cpp \
    imageviewer.cpp \
    softwarecamera.cpp \
    analoginput.cpp \
    FringeFFT.cpp \
    fwhmviewer.cpp \
    motorclass.cpp

HEADERS  += \
    galvocontroller.h \
    galvos.h \
    converter.h \
    daqexception.h \
    config.h \
    camera.h \
    imaqexception.h \
    datasaver.h \
    fringeviewer.h \
    imageviewer.h \
    softwarecamera.h \
    fringeviewer.h \
    analoginput.h \
    float64datasaver.h \
    FringeFFT.h \
    fwhmviewer.h \
    motorclass.h

FORMS    += \
    oct_galvos_form.ui

RESOURCES += \
    icons.qrc

