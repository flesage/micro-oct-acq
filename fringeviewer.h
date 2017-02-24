#ifndef FringeViewer_H
#define FringeViewer_H

#include <QWidget>
#include <QBrush>
#include <QFont>
#include <QPen>
#include <QMutex>
#include "config.h"

class FringeViewer : public QWidget
{
    Q_OBJECT
public:
    explicit FringeViewer(QWidget *parent = 0, int n_alines = 100);
    virtual ~FringeViewer();
    void put(unsigned short* data);
    void Close();
public slots:
    void updateView();

protected:
    void paintEvent(QPaintEvent *event);
private:
    unsigned short int p_fringe[LINE_ARRAY_SIZE];
    unsigned short int p_max_fringe[LINE_ARRAY_SIZE];
    unsigned short int* p_data_buffer;

    int p_nalines;
    QBrush background;
    QFont textFont;
    QPen linePen;
    QPen maxPen;
    QPen textPen;
    QMutex p_mutex;
};

#endif // FringeViewer_H
