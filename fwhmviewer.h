#ifndef FWHMVIEWER_H
#define FWHMVIEWER_H

#include <QWidget>
#include <QBrush>
#include <QFont>
#include <QPen>
#include <QMutex>
#include "config.h"

class FWHMViewer : public QWidget
{
    Q_OBJECT
public:
    explicit FWHMViewer(QWidget *parent = 0, unsigned int data_size = 512);
    virtual ~FWHMViewer();
    void put(unsigned short* data);
    void Close();
signals:

public slots:
    void updateView();
protected:
    void paintEvent(QPaintEvent *event);
private:
    unsigned char* p_aline;
    unsigned int p_data_size;
    QBrush background;
    QFont textFont;
    QPen linePen;
    QPen textPen;
};



#endif // FWHMVIEWER_H
