#ifndef CONVERTER_H
#define CONVERTER_H

#include <QVector>

class Converter
{
public:
    Converter();
    void setScale(double x_um_per_volt,double y_um_per_volt);
    QVector<double>  voltX(QVector<double> um_vect);
    QVector<double>  voltY(QVector<double> um_vect);
    double  voltX(double um_val);
    double  voltY(double um_val);
    void show();
private:
    double p_x_um_per_volt;
    double p_y_um_per_volt;
};

#endif // CONVERTER_H
