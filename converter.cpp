#include "converter.h"

Converter::Converter()
{
    p_x_um_per_volt = 100;
    p_y_um_per_volt = 100;
}

void Converter::setScale(double x_um_per_volt,double y_um_per_volt)
{
    p_x_um_per_volt = x_um_per_volt;
    p_y_um_per_volt = y_um_per_volt;
    return;
}

QVector<double>  Converter::voltX(QVector<double>  um_vect)
{
    QVector<double> val(um_vect.size(),0.0);
    for (int i=0;i < um_vect.size(); i++)
    {
        val[i]=um_vect[i]/p_x_um_per_volt;
    }
    return val;
}

QVector<double>  Converter::voltY(QVector<double>  um_vect)
{
    QVector<double> val(um_vect.size(),0.0);
    for (int i=0;i < um_vect.size(); i++)
    {
        val[i]=um_vect[i]/p_y_um_per_volt;
    }
    return val;
}

double  Converter::voltX(double um_val)
{
    double val;
    val=um_val/p_x_um_per_volt;
    return val;
}

double  Converter::voltY(double um_val)
{
    double val;
    val=um_val/p_y_um_per_volt;
    return val;
}
