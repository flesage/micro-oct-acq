#ifndef OCT3DORTHOGONALVIEWER_H
#define OCT3DORTHOGONALVIEWER_H

#include <QWidget>

namespace Ui {
class oct3dOrthogonalViewer;
}

class oct3dOrthogonalViewer : public QWidget
{
    Q_OBJECT

public:
    explicit oct3dOrthogonalViewer(QWidget *parent = nullptr);
    ~oct3dOrthogonalViewer();

private:
    Ui::oct3dOrthogonalViewer *ui;
};

#endif // OCT3DORTHOGONALVIEWER_H
