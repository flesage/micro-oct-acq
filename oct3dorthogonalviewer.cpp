#include "oct3dorthogonalviewer.h"
#include "ui_oct3dorthogonalviewer.h"

oct3dOrthogonalViewer::oct3dOrthogonalViewer(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::oct3dOrthogonalViewer)
{
    ui->setupUi(this);
}

oct3dOrthogonalViewer::~oct3dOrthogonalViewer()
{
    delete ui;
}
