#include "oct3dorthogonalviewer.h"
#include "ui_oct3dorthogonalviewer.h"
#include <iostream>
#include <cmath>
#include <QPainter>
#include <QPen>
#include <QColor>
#include <QTimer>
#include "config.h"

// TODO: test with a real cameraß

oct3dOrthogonalViewer::oct3dOrthogonalViewer(QWidget *parent, int nx, int n_extra, int ny, int nz) :
    QWidget(parent),
    ui(new Ui::oct3dOrthogonalViewer),
    f_fft(1, 1) // TODO: use n_repeat and factor for the f_fft constructor
{
    ui_is_ready = false;
    ui->setupUi(this);

    // Setup Data
    p_nx = nx;
    p_ny = ny;
    p_nz = nz;
    p_n_extra = n_extra;
    p_line_thickness = 0.025;
    p_x_line_thickness = int(p_line_thickness * p_nx);
    p_y_line_thickness = int(p_line_thickness * p_ny);
    p_z_line_thickness = int(p_line_thickness * p_nz);
    p_projection_mode = AVERAGE;
    p_log_transform = true;
    p_overlay = true;

    // Prepare the data reconstruction
    float dimz = 3.5; // FIXME: dummy axial resolution
    float dimx = 3.0; // FIXME: dummy lateral resolution
    f_fft.init(LINE_ARRAY_SIZE, p_nx + p_n_extra, dimz, dimx);
    p_data_buffer = new unsigned short[(p_nx + p_n_extra)*LINE_ARRAY_SIZE];
    p_image_buffer = new unsigned char[(p_nx + p_n_extra)*p_nz];
    p_oct_buffer = af::constant(0.0, p_nz, p_nx, p_ny, f32);

    // Prepare the images
    p_image_xy = QImage(p_nx, p_ny, QImage::Format_Indexed8);
    p_image_xz = QImage(p_nx, p_nz, QImage::Format_Indexed8);
    p_image_yz = QImage(p_nz, p_ny, QImage::Format_Indexed8);

    // Adapt UI to volume size
    ui->horizontalSlider_x->setRange(0, p_nx-1);
    ui->horizontalSlider_y->setRange(0, p_ny-1);
    ui->horizontalSlider_z->setRange(0, p_nz-1);
    ui->horizontalSlider_xRange->setRange(1, p_nx-1);
    ui->horizontalSlider_xRange->setValue(p_nx-1);
    ui->horizontalSlider_yRange->setRange(1, p_ny-1);
    ui->horizontalSlider_yRange->setValue(p_ny-1);
    ui->horizontalSlider_zRange->setRange(1, p_nz-1);
    ui->horizontalSlider_zRange->setValue(p_nz-1);
    ui->comboBox_projectionType->addItem("Average");
    ui->comboBox_projectionType->addItem("Maximum");
    ui->comboBox_projectionType->addItem("Minimum");
    ui->comboBox_projectionType->addItem("Variance");
    set_x(p_nx/2);
    set_y(p_ny/2);
    set_z(p_nz/2);
    set_slice_thickness(p_nz * 0.05);
    p_current_frame = 0;

    // Signals
    connect(ui->horizontalSlider_x, SIGNAL(valueChanged(int)), this, SLOT(set_x(int)));
    connect(ui->horizontalSlider_y, SIGNAL(valueChanged(int)), this, SLOT(set_y(int)));
    connect(ui->horizontalSlider_z, SIGNAL(valueChanged(int)), this, SLOT(set_z(int)));
    connect(ui->horizontalSlider_xRange, SIGNAL(valueChanged(int)), this, SLOT(slot_update_view()));
    connect(ui->horizontalSlider_yRange, SIGNAL(valueChanged(int)), this, SLOT(slot_update_view()));
    connect(ui->horizontalSlider_zRange, SIGNAL(valueChanged(int)), this, SLOT(slot_update_view()));
    connect(ui->comboBox_projectionType, SIGNAL(currentIndexChanged(int)), this, SLOT(set_projection_mode(int)));
    connect(ui->checkBox_logTransform, SIGNAL(stateChanged(int)), this, SLOT(set_log_transform(int)));
    connect(ui->checkBox_xyzOverlay, SIGNAL(stateChanged(int)), this, SLOT(set_overlay(int)));
    connect(ui->checkBox_showBscan, SIGNAL(stateChanged(int)), this, SLOT(slot_update_view()));
    connect(ui->checkBox_showRange, SIGNAL(stateChanged(int)), this, SLOT(slot_update_view()));

    ui_is_ready = true;
}

oct3dOrthogonalViewer::~oct3dOrthogonalViewer()
{
    delete ui;
}

void oct3dOrthogonalViewer::put(unsigned short* fringe, unsigned int frame_number) {
    double p_image_threshold = 0.05; // TODO: read this from the interface
    int p_hanning_threshold = 100; // TODO: read this from the interface
    if (p_mutex.tryLock())
    {
        memcpy(p_data_buffer, fringe, (p_nx+p_n_extra)*LINE_ARRAY_SIZE*sizeof(unsigned short));
        f_fft.interp_and_do_fft(p_data_buffer, p_image_buffer, p_image_threshold, p_hanning_threshold);
        af::array bscan = af::array(p_nz, p_nx + p_n_extra, p_image_buffer, afHost).as(f32);
        bscan = bscan(af::span, af::seq(p_nx));
        p_current_frame = frame_number % p_ny;
        p_oct_buffer(af::span, af::span, p_current_frame%p_ny) =  bscan(af::span, af::span, 0);
        p_mutex.unlock();
    }
}

void oct3dOrthogonalViewer::set_x(int x)
{
    p_current_x = x;

    // Update UI
    ui->horizontalSlider_x->setValue(x);
    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::set_y(int y)
{
    p_current_y = y;

    // Update UI
    ui->horizontalSlider_y->setValue(y);
    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::set_z(int z)
{
    p_current_z = z;

    // Update UI
    ui->horizontalSlider_z->setValue(z);
    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::set_slice_thickness(int thickness)
{
    p_slice_thickness = thickness;

    // Update UI
    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::set_projection_mode(int mode)
{
    p_projection_mode = mode;

    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::set_log_transform(int value)
{
    if (value == 0) {
        p_log_transform = false;
    } else{
        p_log_transform = true;
    }

    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::set_overlay(int value)
{
    if (value == 0) {
        p_overlay = false;
    } else {
        p_overlay = true;
    }

    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::slot_update_view()
{
    p_mutex.lock();
    p_oct = p_oct_buffer.copy();
    p_mutex.unlock();

    // Computing X projection
    int rx = ui->horizontalSlider_xRange->sliderPosition() / 2;
    int x_min = p_current_x - rx;
    x_min = x_min > 0 ? x_min : 0;
    int x_max = p_current_x + rx;
    x_max = x_max < p_nx-1 ? x_max : p_nx - 1;

    af::array mip_x;
    switch (p_projection_mode) {
    case AVERAGE:
        mip_x = af::mean(p_oct(af::span, af::seq(x_min, x_max), af::span), 1);
        break;
    case MAXIMUM:
        mip_x = af::max(p_oct(af::span, af::seq(x_min, x_max), af::span), 1);
        break;
    case MINIMUM:
        mip_x = af::min(p_oct(af::span, af::seq(x_min, x_max), af::span), 1);
        break;
    case VARIANCE:
        mip_x = af::var(p_oct(af::span, af::seq(x_min, x_max), af::span), AF_VARIANCE_DEFAULT, 1);
        break;
    default:
        mip_x = af::mean(p_oct(af::span, af::seq(x_min, x_max), af::span), 1);
        break;
    }

    // Computing Y projection
    int ry = ui->horizontalSlider_yRange->sliderPosition() / 2;
    int y_min = p_current_y - ry;
    y_min = y_min > 0 ? y_min : 0;
    int y_max = p_current_y + ry;
    y_max = y_max < p_ny-1 ? y_max : p_ny - 1;
    af::array mip_y;
    switch (p_projection_mode) {
    case AVERAGE:
        mip_y = af::mean(p_oct(af::span, af::span, af::seq(y_min, y_max)), 2);
        break;
    case MAXIMUM:
        mip_y = af::max(p_oct(af::span, af::span, af::seq(y_min, y_max)), 2);
        break;
    case MINIMUM:
        mip_y = af::min(p_oct(af::span, af::span, af::seq(y_min, y_max)), 2);
        break;
    case VARIANCE:
        mip_y = af::var(p_oct(af::span, af::span, af::seq(y_min, y_max)), AF_VARIANCE_DEFAULT, 2);
        break;
    default:
        mip_y = af::mean(p_oct(af::span, af::span, af::seq(y_min, y_max)), 2);
        break;
    }
    mip_y = af::transpose(mip_y);

    // Computing Z projection
    int rz = ui->horizontalSlider_zRange->sliderPosition() / 2;
    int z_min = p_current_z - rz;
    z_min = z_min > 0 ? z_min : 0;
    int z_max = p_current_z + rz;
    z_max = z_max < p_nz-1 ? z_max : p_nz - 1;
    af::array mip_z;
    switch (p_projection_mode) {
    case AVERAGE:
        mip_z = af::mean(p_oct(af::seq(z_min, z_max), af::span, af::span), 0);
        break;
    case MAXIMUM:
        mip_z = af::max(p_oct(af::seq(z_min, z_max), af::span, af::span), 0);
        break;
    case MINIMUM:
        mip_z = af::min(p_oct(af::seq(z_min, z_max), af::span, af::span), 0);
        break;
    case VARIANCE:
        mip_z = af::var(p_oct(af::seq(z_min, z_max), af::span, af::span), AF_VARIANCE_DEFAULT, 0);
        break;
    default:
        mip_z = af::mean(p_oct(af::seq(z_min, z_max), af::span, af::span), 0);
        break;
    }

    // Contrast adjustment and filtering
    if (p_log_transform) {
        mip_x = af::log(mip_x + 0.001);
        mip_y = af::log(mip_y + 0.001);
        mip_z = af::log(mip_z + 0.001);
    }

    // Adjusting contrast
    float l_max_x = af::max<float>(mip_x);
    float l_min_x = af::min<float>(mip_x);
    mip_x = 255.0 * (mip_x - l_min_x) / (l_max_x - l_min_x);
    mip_x.as(u8).host(p_image_yz.bits());
    QImage tmp_x = p_image_yz.convertToFormat(QImage::Format_ARGB32);

    float l_max_y = af::max<float>(mip_y);
    float l_min_y = af::min<float>(mip_y);
    mip_y = 255.0 * (mip_y - l_min_y) / (l_max_y - l_min_y);
    mip_y.as(u8).host(p_image_xz.bits());
    QImage tmp_y = p_image_xz.convertToFormat(QImage::Format_ARGB32);

    float l_max_z = af::max<float>(mip_z);
    float l_min_z = af::min<float>(mip_z);
    mip_z = 255.0 * (mip_z - l_min_z) / (l_max_z - l_min_z);
    mip_z.as(u8).host(p_image_xy.bits());
    QImage tmp_z = p_image_xy.convertToFormat(QImage::Format_ARGB32);

    // Drawing annotations
    pix_xy = QPixmap::fromImage(tmp_z);
    if (ui->checkBox_showRange->isChecked()) // Range display
    {
        QPainter painter_xy = QPainter(&pix_xy);
        painter_xy.setPen(QPen(QColor(255,0,0,64)));
        painter_xy.setBrush(QBrush(QColor(255,0,0,32)));
        painter_xy.drawRect(x_min, 0, (x_max - x_min), p_ny);

        painter_xy.setPen(QPen(QColor(0,255,0,64)));
        painter_xy.setBrush(QBrush(QColor(0,255,0,32)));
        painter_xy.drawRect(0, y_min, p_nx, (y_max - y_min));
    }

    if (p_overlay == true){
        QPainter painter_xy = QPainter(&pix_xy);
        painter_xy.setPen(QPen(QColor(255,0,0,128), p_x_line_thickness));
        painter_xy.drawLine(p_current_x, 0, p_current_x, p_ny-1);
        painter_xy.setPen(QPen(QColor(0,255,0,128), p_y_line_thickness));
        painter_xy.drawLine(0, p_current_y, p_nx-1, p_current_y);

        // Border
        painter_xy.setPen(QPen(QColor(0,0,255,128), p_x_line_thickness));
        painter_xy.drawLine(0, 0, 0, p_ny-1);
        painter_xy.drawLine(p_nx-1, 0, p_nx-1, p_ny-1);
        painter_xy.setPen(QPen(QColor(0,0,255,128), p_y_line_thickness));
        painter_xy.drawLine(0, 0, p_nx-1, 0);
        painter_xy.drawLine(0, p_ny-1, p_nx-1, p_ny-1);

    }
    if (ui->checkBox_showBscan->isChecked()){
        QPainter painter_xy = QPainter(&pix_xy);
        painter_xy.setPen(QPen(QColor(255, 255, 255, 128), p_y_line_thickness));
        painter_xy.drawLine(0, p_current_frame, p_nx-1, p_current_frame);
    }

    pix_xz = QPixmap::fromImage(tmp_y);
    if (ui->checkBox_showRange->isChecked()) // Range display
    {
        QPainter painter_xz = QPainter(&pix_xz);
        painter_xz.setPen(QPen(QColor(255,0,0,64)));
        painter_xz.setBrush(QBrush(QColor(255,0,0,32)));
        painter_xz.drawRect(x_min, 0, (x_max - x_min), p_nz);
        painter_xz.setPen(QPen(QColor(0,0,255,64)));
        painter_xz.setBrush(QBrush(QColor(0,0,255,32)));
        painter_xz.drawRect(0, z_min, p_nx, (z_max - z_min));
    }
    if (p_overlay == true){
        QPainter painter_xz = QPainter(&pix_xz);
        painter_xz.setPen(QPen(QColor(255,0,0,128), p_x_line_thickness));
        painter_xz.drawLine(p_current_x, 0, p_current_x, p_nz-1);
        painter_xz.setPen(QPen(QColor(0,0,255,128), p_z_line_thickness));
        painter_xz.drawLine(0, p_current_z, p_nx-1, p_current_z);

        // Border
        painter_xz.setPen(QPen(QColor(0,255,0,128), p_z_line_thickness));
        painter_xz.drawLine(0, 0, p_nx-1, 0);
        painter_xz.drawLine(0, p_nz-1, p_nx-1, p_nz-1);

        painter_xz.setPen(QPen(QColor(0,255,0,128), p_x_line_thickness));
        painter_xz.drawLine(0, 0, 0, p_nz-1);
        painter_xz.drawLine(p_nx-1, 0, p_nx-1, p_nz-1);
    }

    pix_yz = QPixmap::fromImage(tmp_x);

    if (ui->checkBox_showRange->isChecked()) // Range display
    {
        QPainter painter_yz = QPainter(&pix_yz);
        painter_yz.setPen(QPen(QColor(0,255,0,64)));
        painter_yz.setBrush(QBrush(QColor(0,255,0,32)));
        painter_yz.drawRect(0, y_min, p_nz, (y_max-y_min));
        painter_yz.setPen(QPen(QColor(0,0,255,64)));
        painter_yz.setBrush(QBrush(QColor(0,0,255,32)));
        painter_yz.drawRect(z_min, 0, (z_max - z_min), p_ny);
    }

    if (p_overlay == true){
        QPainter painter_yz = QPainter(&pix_yz);
        painter_yz.setPen(QPen(QColor(0,0,255,128), p_z_line_thickness));
        painter_yz.drawLine(p_current_z, 0, p_current_z, p_ny);
        painter_yz.setPen(QPen(QColor(0,255,0,128), p_y_line_thickness));
        painter_yz.drawLine(0, p_current_y, p_nz, p_current_y);

        // Border
        painter_yz.setPen(QPen(QColor(255,0,0,128), p_y_line_thickness));
        painter_yz.drawLine(0, 0, p_nz-1, 0);
        painter_yz.drawLine(0, p_ny-1, p_nz-1, p_ny-1);
        painter_yz.setPen(QPen(QColor(255,0,0,128), p_z_line_thickness));
        painter_yz.drawLine(0, 0, 0, p_ny-1);
        painter_yz.drawLine(p_nz-1, 0, p_nz-1, p_ny-1);
    }

    if (ui->checkBox_showBscan->isChecked()){
        QPainter painter_yz = QPainter(&pix_yz);
        painter_yz.setPen(QPen(QColor(255, 255, 255, 128), p_y_line_thickness));
        painter_yz.drawLine(0, p_current_frame, p_nz-1, p_current_frame);
    }

    // Set as pixmaps
    ui->label_yz->setPixmap(pix_yz.scaled(ui->label_yz->size(),
                                       Qt::IgnoreAspectRatio, Qt::FastTransformation));
    ui->label_xz->setPixmap(pix_xz.scaled(ui->label_xz->size(),
                                       Qt::IgnoreAspectRatio, Qt::FastTransformation));
    ui->label_xy->setPixmap(pix_xy.scaled(ui->label_xy->size(),
                                       Qt::IgnoreAspectRatio, Qt::FastTransformation));
}

void oct3dOrthogonalViewer::resizeEvent(QResizeEvent *)
{
    if(!pix_xy.isNull()){
        ui->label_xy->setPixmap(pix_xy.scaled(ui->label_xy->size(),
                                           Qt::IgnoreAspectRatio, Qt::FastTransformation));
    }
    if (!pix_xz.isNull()){
        ui->label_xz->setPixmap(pix_xz.scaled(ui->label_xz->size(),
                                           Qt::IgnoreAspectRatio, Qt::FastTransformation));
    }
    if (!pix_yz.isNull()){
        ui->label_yz->setPixmap(pix_yz.scaled(ui->label_yz->size(),
                                           Qt::IgnoreAspectRatio, Qt::FastTransformation));
    }
}
