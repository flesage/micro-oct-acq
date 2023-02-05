#include "oct3dorthogonalviewer.h"
#include "ui_oct3dorthogonalviewer.h"
#include <iostream>
#include <cmath>
#include <QPainter>
#include <QPen>
#include <QColor>
#include <QTimer>

// TODO: Add mouse and keyboard interactions (scroll and mouse click position uptdate)
// TODO: Show the projection range with the overlay
// TODO: Add colormap options
//  https://stackoverflow.com/questions/41744654/is-there-a-default-color-table-colormap-available-in-qt
//  http://www.kennethmoreland.com/color-advice/
// TODO: add option to perform averaging?
// TODO: add simulation mode

oct3dOrthogonalViewer::oct3dOrthogonalViewer(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::oct3dOrthogonalViewer)
{
    ui_is_ready = false;
    ui->setupUi(this);

    // Setup Data
    p_nx = 256;
    p_ny = 256;
    p_nz = 128;
    p_line_thickness = 3;
    p_projection_mode = AVERAGE;
    p_log_transform = true;
    p_overlay = true;

    // Prepare the pen
    pen_x = QPen(QColor(255, 0, 0, 128), p_line_thickness); // x = red
    pen_y = QPen(QColor(0, 255, 0, 128), p_line_thickness); // y = green
    pen_z = QPen(QColor(0, 0, 255, 128), p_line_thickness); // z = blue

    //p_oct = af::array(p_nz, p_nx, p_ny, f32);

    // Simulating an OCT volume
    p_oct = af::randu(p_nz, p_nx, p_ny, f32);
    for (int z=0; z<p_nz; z++){
        p_oct(z,af::span,af::span) = p_oct(z,af::span,af::span) * exp(-z * 0.1);
    }
    p_image_xy = QImage(p_nx, p_ny, QImage::Format_Indexed8);
    p_image_xz = QImage(p_nx, p_nz, QImage::Format_Indexed8);
    p_image_yz = QImage(p_nz, p_ny, QImage::Format_Indexed8);
    simulation_timer = new QTimer();
    p_y_sim = 0;

    // Adapt UI to volume size
    ui->horizontalSlider_x->setRange(0, p_nx-1);
    ui->spinBox_x->setRange(0, p_nx-1);
    ui->horizontalSlider_y->setRange(0, p_ny-1);
    ui->spinBox_y->setRange(0, p_ny-1);
    ui->horizontalSlider_z->setRange(0, p_nz-1);
    ui->spinBox_z->setRange(0, p_nz-1);
    ui->spinBox_sliceThickness->setRange(1, std::max(std::min(p_nx, p_ny), p_nz));
    ui->comboBox_projectionType->addItem("Average");
    ui->comboBox_projectionType->addItem("Maximum");
    ui->comboBox_projectionType->addItem("Minimum");
    ui->comboBox_projectionType->addItem("Variance");
    set_x(p_nx/2);
    set_y(p_ny/2);
    set_z(p_nz/2);
    set_slice_thickness(p_nz * 0.05);


    // Signals
    connect(ui->horizontalSlider_x, SIGNAL(valueChanged(int)), this, SLOT(set_x(int)));
    connect(ui->spinBox_x, SIGNAL(valueChanged(int)), this, SLOT(set_x(int)));
    connect(ui->horizontalSlider_y, SIGNAL(valueChanged(int)), this, SLOT(set_y(int)));
    connect(ui->spinBox_y, SIGNAL(valueChanged(int)), this, SLOT(set_y(int)));
    connect(ui->horizontalSlider_z, SIGNAL(valueChanged(int)), this, SLOT(set_z(int)));
    connect(ui->spinBox_z, SIGNAL(valueChanged(int)), this, SLOT(set_z(int)));
    connect(ui->spinBox_sliceThickness, SIGNAL(valueChanged(int)), this, SLOT(set_slice_thickness(int)));
    connect(ui->comboBox_projectionType, SIGNAL(currentIndexChanged(int)), this, SLOT(set_projection_mode(int)));
    connect(ui->checkBox_logTransform, SIGNAL(stateChanged(int)), this, SLOT(set_log_transform(int)));
    connect(ui->checkBox_xyzOverlay, SIGNAL(stateChanged(int)), this, SLOT(set_overlay(int)));
    connect(simulation_timer, SIGNAL(timeout()), this, SLOT(simulate_bscan()));

    ui_is_ready = true;
    simulation_timer->start(100);
}

oct3dOrthogonalViewer::~oct3dOrthogonalViewer()
{
    simulation_timer -> stop();
    delete ui;
}

// Replace a b-scan given a y frame number.
void oct3dOrthogonalViewer::put(const af::array& data, unsigned int frame_number)
{
    p_oct(af::span, af::span, frame_number) = data;
    slot_update_view();
}

void oct3dOrthogonalViewer::set_x(int x)
{
    std::cout << "Setting x=" << x << std::endl;
    p_current_x = x;

    // Update UI
    ui->horizontalSlider_x->setValue(x);
    ui->spinBox_x->setValue(x);
    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::set_y(int y)
{
    std::cout << "Setting y=" << y << std::endl;
    p_current_y = y;

    // Update UI
    ui->horizontalSlider_y->setValue(y);
    ui->spinBox_y->setValue(y);
    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::set_z(int z)
{
    std::cout << "Setting z=" << z << std::endl;
    p_current_z = z;

    // Update UI
    ui->horizontalSlider_z->setValue(z);
    ui->spinBox_z->setValue(z);
    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::set_slice_thickness(int thickness)
{
    std::cout << "Setting slice_thickness=" << thickness << std::endl;
    p_slice_thickness = thickness;

    // Update UI
    ui->spinBox_sliceThickness->setValue(p_slice_thickness);
    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::set_projection_mode(int mode)
{
    std::cout << "Setting projection_mode=" << mode << std::endl;
    p_projection_mode = mode;

    if (ui_is_ready){
        slot_update_view();
    }
}

void oct3dOrthogonalViewer::set_log_transform(int value)
{
    std::cout << "Setting log_transform=" << value << std::endl;
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
    std::cout << "Setting overlay=" << value << std::endl;
    if (value == 0) {
        p_overlay = false;
    } else {
        p_overlay = true;
    }

    if (ui_is_ready){
        slot_update_view();
    }
}

// TODO: There is a bug with the bscan displays, every other row or column is missing.
void oct3dOrthogonalViewer::slot_update_view()
{
    std::cout << "Updating the view" << std::endl;

    // Computing X projection
    // TODO: Change the behaviour of the slice_thickness to be centered.
    int n_slices_x;
    if (p_current_x + p_slice_thickness > p_nx - 1)
    {
        n_slices_x = p_nx - p_current_x;
    }
    else {
        n_slices_x = p_slice_thickness;
    }
    af::array mip_x;
    switch (p_projection_mode) {
    case AVERAGE:
        mip_x = af::mean(p_oct(af::span, af::seq(p_current_x, p_current_x + n_slices_x-1), af::span), 1);
        break;
    case MAXIMUM:
        mip_x = af::max(p_oct(af::span, af::seq(p_current_x, p_current_x + n_slices_x-1), af::span), 1);
        break;
    case MINIMUM:
        mip_x = af::min(p_oct(af::span, af::seq(p_current_x, p_current_x + n_slices_x-1), af::span), 1);
        break;
    case VARIANCE:
        mip_x = af::var(p_oct(af::span, af::seq(p_current_x, p_current_x + n_slices_x-1), af::span), AF_VARIANCE_DEFAULT, 1);
        break;
    default:
        mip_x = af::mean(p_oct(af::span, af::seq(p_current_x, p_current_x + n_slices_x-1), af::span), 1);
        break;
    }

    // Computing Y projection
    int n_slices_y;
    if (p_current_y + p_slice_thickness > p_ny - 1)
    {
        n_slices_y = p_ny - p_current_y;
    }
    else {
        n_slices_y = p_slice_thickness;
    }
    af::array mip_y;
    switch (p_projection_mode) {
    case AVERAGE:
        mip_y = af::mean(p_oct(af::span, af::span, af::seq(p_current_y, p_current_y + n_slices_y-1)), 2);
        break;
    case MAXIMUM:
        mip_y = af::max(p_oct(af::span, af::span, af::seq(p_current_y, p_current_y + n_slices_y-1)), 2);
        break;
    case MINIMUM:
        mip_y = af::min(p_oct(af::span, af::span, af::seq(p_current_y, p_current_y + n_slices_y-1)), 2);
        break;
    case VARIANCE:
        mip_y = af::var(p_oct(af::span, af::span, af::seq(p_current_y, p_current_y + n_slices_y-1)), AF_VARIANCE_DEFAULT, 2);
        break;
    default:
        mip_y = af::mean(p_oct(af::span, af::span, af::seq(p_current_y, p_current_y + n_slices_y-1)), 2);
        break;
    }
    mip_y = af::transpose(mip_y);

    // Computing Z projection
    int n_slices_z;
    if (p_current_z + p_slice_thickness > p_nz - 1)
    {
        n_slices_z = p_nz - p_current_z;
    }
    else {
        n_slices_z = p_slice_thickness;
    }
    af::array mip_z;
    switch (p_projection_mode) {
    case AVERAGE:
        mip_z = af::mean(p_oct(af::seq(p_current_z, p_current_z + n_slices_z-1), af::span, af::span), 0);
        break;
    case MAXIMUM:
        mip_z = af::max(p_oct(af::seq(p_current_z, p_current_z + n_slices_z-1), af::span, af::span), 0);
        break;
    case MINIMUM:
        mip_z = af::min(p_oct(af::seq(p_current_z, p_current_z + n_slices_z-1), af::span, af::span), 0);
        break;
    case VARIANCE:
        mip_z = af::var(p_oct(af::seq(p_current_z, p_current_z + n_slices_z-1), af::span, af::span), AF_VARIANCE_DEFAULT, 0);
        break;
    default:
        mip_z = af::mean(p_oct(af::seq(p_current_z, p_current_z + n_slices_z-1), af::span, af::span), 0);
        break;
    }

    // TODO: Contrast adjustment and filtering
    if (p_log_transform) {
        mip_x = af::log(mip_x + 0.001);
        mip_y = af::log(mip_y + 0.001);
        mip_z = af::log(mip_z + 0.001);
    }

    // Adjusting contrast
    float l_max = std::max(af::max<float>(mip_x), af::max<float>(mip_y));
    l_max = std::max(l_max, af::max<float>(mip_z));

    float l_min = std::min(af::min<float>(mip_x), af::min<float>(mip_y));
    l_min = std::min(l_min, af::min<float>(mip_z));

    mip_x = 255.0 * (mip_x - l_min) / (l_max - l_min);
    mip_x.as(u8).host(p_image_yz.bits());
    QImage tmp_x = p_image_yz.convertToFormat(QImage::Format_ARGB32);

    mip_y = 255.0 * (mip_y - l_min) / (l_max - l_min);
    mip_y.as(u8).host(p_image_xz.bits());
    QImage tmp_y = p_image_xz.convertToFormat(QImage::Format_ARGB32);

    mip_z = 255.0 * (mip_z - l_min) / (l_max - l_min);
    mip_z.as(u8).host(p_image_xy.bits());
    QImage tmp_z = p_image_xy.convertToFormat(QImage::Format_ARGB32);

    // Drawing annotations
    pix_xy = QPixmap::fromImage(tmp_z);
    if (p_overlay == true){
        QPainter painter_xy = QPainter(&pix_xy);
        painter_xy.setPen(pen_x);
        painter_xy.drawLine(p_current_x, 0, p_current_x, p_ny);
        painter_xy.setPen(pen_y);
        painter_xy.drawLine(0, p_current_y, p_nx, p_current_y);
        painter_xy.setPen(pen_z);
        painter_xy.drawRect(0, 0, p_nx-1, p_ny-1);
    }

    pix_xz = QPixmap::fromImage(tmp_y);
    if (p_overlay == true){
        QPainter painter_xz = QPainter(&pix_xz);
        painter_xz.setPen(pen_x);
        painter_xz.drawLine(p_current_x, 0, p_current_x, p_nz);
        painter_xz.setPen(pen_z);
        painter_xz.drawLine(0, p_current_z, p_nx, p_current_z);
        painter_xz.setPen(pen_y);
        painter_xz.drawRect(0, 0, p_nx-1, p_nz-1);
    }

    pix_yz = QPixmap::fromImage(tmp_x);
    if (p_overlay == true){
        QPainter painter_yz = QPainter(&pix_yz);
        painter_yz.setPen(pen_z);
        painter_yz.drawLine(p_current_z, 0, p_current_z, p_ny);
        painter_yz.setPen(pen_y);
        painter_yz.drawLine(0, p_current_y, p_nz, p_current_y);
        painter_yz.setPen(pen_x);
        painter_yz.drawRect(0, 0, p_nz-1, p_ny-1);
    }


    // Set as pixmaps
    // TODO: Check the image's orientations
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

void oct3dOrthogonalViewer::simulate_bscan(){
    std::cout << "Simulating a b-scan for frame y=" << p_y_sim << std::endl;
    simulated_bscan = af::randu(p_nz, p_nx, f32);
    for (int z=0; z<p_nz; z++){
        simulated_bscan(z, af::span) = simulated_bscan(z, af::span) * exp(-z * 0.1);
    }
    put(simulated_bscan, p_y_sim);

    // Updating the simulated frame
    p_y_sim++;
    if (p_y_sim == p_ny){
        p_y_sim = 0;
    }
}
