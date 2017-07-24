#-------------------------------------------------
#
# Project created by QtCreator 2016-03-31T18:23:47
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = oct
TEMPLATE = app
RC_FILE = oct.rc

CUDA_SOURCES += fringe_norm.cu
CUDA_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0"            # Path to cuda toolkit install
SYSTEM_NAME = x64         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = compute_61           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math


# Add the necessary libraries
INCLUDEPATH += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include"
INCLUDEPATH += "C:\Program Files (x86)\National Instruments\NI-DAQ\DAQmx ANSI C Dev\include"
INCLUDEPATH += "C:\Program Files (x86)\National Instruments\NI-IMAQ\Include"

LIBS += -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64" -lcudart -lcusparse -lcufft -lnppi -lcublas
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
    FringeFFT.cpp

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
    fringe_norm.h \
    FringeFFT.h

FORMS    += \
    oct_galvos_form.ui

RESOURCES += \
    icons.qrc

# This makes the .cu files appear in your project
OTHER_FILES +=  fringe_norm.cu

# The following library conflicts with something in Cuda
MSVCRT_LINK_FLAG_DEBUG = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = ${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC  --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
    --compile -cudart static -g -DWIN32 -D_MBCS \
    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
    -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = ${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
    --compile -cudart static -DWIN32 -D_MBCS \
    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
