#include "galvocontroller.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    GalvoController w;
    w.show();
    w.resize(555,855);
    //w.setFixedSize(685,820);

    return a.exec();
}
