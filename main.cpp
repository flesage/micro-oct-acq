#include "galvocontroller.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    GalvoController w;
    w.show();

    return a.exec();
}
