#include "oct3dorthogonalviewer.h"
#include "ui_oct3dorthogonalviewer.h"
#include <iostream>

oct3dOrthogonalViewer::oct3dOrthogonalViewer(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::oct3dOrthogonalViewer)
{
    ui_is_ready = false;
    ui->setupUi(this);

    // Setup Data
    p_nx = 32;
    p_ny = 64;
    p_nz = 128;
    p_projection_mode = AVERAGE;

    //p_oct = af::array(p_nz, p_nx, p_ny, f32);
    p_oct = af::randu(p_nz, p_nx, p_ny, f32);
    p_image_xy = QImage(p_nx, p_ny, QImage::Format_Indexed8);
    p_image_xz = QImage(p_nx, p_nz, QImage::Format_Indexed8);
    p_image_yz = QImage(p_ny, p_nz, QImage::Format_Indexed8);

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

    ui_is_ready = true;
}

oct3dOrthogonalViewer::~oct3dOrthogonalViewer()
{
    delete ui;
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

void oct3dOrthogonalViewer::slot_update_view()
{
    std::cout << "Updating the view" << std::endl;

    // Computing X projection
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

    // Filter the projections

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
    // TODO

    // Set as pixmaps
    pix = QPixmap::fromImage(tmp_x);
    ui->label_yz->setPixmap(pix.scaled(ui->label_yz->size(),
                                       Qt::IgnoreAspectRatio, Qt::SmoothTransformation));


    pix = QPixmap::fromImage(tmp_y);
    ui->label_xz->setPixmap(pix.scaled(ui->label_xz->size(),
                                       Qt::IgnoreAspectRatio, Qt::SmoothTransformation));

    pix = QPixmap::fromImage(tmp_z);
    ui->label_xy->setPixmap(pix.scaled(ui->label_xy->size(),
                                       Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
}
