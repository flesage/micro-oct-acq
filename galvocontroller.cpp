#include "galvocontroller.h"
#include <iostream>
#include <QIntValidator>
#include <QDoubleValidator>
#include <QDir>
#include <QFileDialog>
#include <QTimer>
#include <QStringList>
#include <QTextStream>
#include <QMessageBox>
#include "ui_oct_galvos_form.h"
#include "config.h"
#include <QApplication>
#include "motorclass.h"
#include "piezostage.h"

GalvoController::GalvoController() :
    ui(new Ui::OCTGalvosForm), p_galvos(GALVOS_DEV,GALVOS_AOX,GALVOS_AOY), p_settings("Polytechnique/LIOM","OCT"),
    p_ai(0), p_ai_data_saver(0)
{
    ui->setupUi(this);
    p_finite_acquisition = false;
    p_stack_acquisition = false;
    p_acq_index = 0;
    p_center_x=0.0;
    p_center_y=0.0;
    p_offset_x=0.0;
    p_offset_y=0.0;
    p_coeff_x=1.0;
    p_coeff_y_quad=1.0;
    p_coeff_x_quad=1.0;
    p_coeff_xy=1.0;
    p_coeff_yx=1.0;
    p_coeff_xy_quad=1.0;
    p_coeff_yx_quad=1.0;
    p_coeff_yyx=1.0;
    p_coeff_xxy=1.0;

    p_offset_x_added=0.0;
    p_offset_y_added=0.0;

    p_block_size = 256;
    p_camera = 0;
    p_fringe_view = 0;
    p_image_view = 0;
    p_data_saver = 0;
    p_start_line_x = 0;
    p_stop_line_x = 0;
    p_start_line_y = 0;
    p_stop_line_y = 0;
    p_line_length = 0.0;

    p_start_viewline = ui->lineEdit_startLine->text().toInt();
    p_stop_viewline = ui->lineEdit_stopLine->text().toInt();

    motors = new MotorClass();

    double radians_per_volt = 2*3.14159265359/(360*0.8);
    double f1=50.0;
    double f2=100.0;
    double fobj=18.0;
    double scale_um_per_volt=(2*fobj*f1/f2*radians_per_volt)*1000.0;
    Converter unit_converter;
    unit_converter.setScale(scale_um_per_volt,scale_um_per_volt);
    p_galvos.setUnitConverter(unit_converter);

    QIntValidator* validator=new QIntValidator(1,4096,this);
    ui->lineEdit_nx->setValidator(validator);
    ui->lineEdit_ny->setValidator(validator);
    ui->lineEdit_width->setValidator(validator);
    ui->lineEdit_height->setValidator(validator);
    QIntValidator* validator2=new QIntValidator(2,200,this);
    ui->lineEdit_extrapoints->setValidator(validator2);
    QIntValidator* validator3=new QIntValidator(1,128,this);
    ui->lineEdit_fastaxisrepeat->setValidator(validator3);
    QIntValidator* validator4=new QIntValidator(1,1000,this);
    ui->lineEdit_aline_repeat->setValidator(validator4);
    QIntValidator* validator5=new QIntValidator(10,2048,this);
    ui->lineEdit_displayPoints->setValidator(validator5);

    connect(ui->lineEdit_linerate,SIGNAL(editingFinished()),this,SLOT(updateInfo()));
    connect(ui->lineEdit_nx,SIGNAL(editingFinished()),this,SLOT(updateInfo()));
    connect(ui->lineEdit_extrapoints,SIGNAL(editingFinished()),this,SLOT(updateInfo()));
    connect(ui->lineEdit_exposure,SIGNAL(editingFinished()),this,SLOT(updateInfo()));
    connect(ui->lineEdit_width,SIGNAL(editingFinished()),this,SLOT(updateInfo()));
    connect(ui->lineEdit_aline_repeat,SIGNAL(editingFinished()),this,SLOT(updateInfo()));

    connect(ui->pushButton_piezoOn,SIGNAL(clicked()),this,SLOT(turnPiezoOn()));
    connect(ui->pushButton_piezoOff,SIGNAL(clicked()),this,SLOT(turnPiezoOff()));
    connect(ui->pushButton_piezoHome,SIGNAL(clicked()),this,SLOT(homePiezo()));
    connect(ui->spinBox_piezoSpeed,SIGNAL(valueChanged(int)),this,SLOT(updateSpeedPiezo()));
    connect(ui->pushButton_piezoJog,SIGNAL(clicked()),this,SLOT(jogPiezo()));
    connect(ui->pushButton_piezoStop,SIGNAL(clicked()),this,SLOT(stopPiezo()));

    connect(ui->lineEdit_startLine,SIGNAL(editingFinished()),this,SLOT(slot_updateViewLinePositions()));
    connect(ui->lineEdit_stopLine,SIGNAL(editingFinished()),this,SLOT(slot_updateViewLinePositions()));
    connect(ui->checkBox_view_line,SIGNAL(clicked(bool)),this,SLOT(slot_updateViewLinePositions()));

    connect(ui->comboBox_scantype,SIGNAL(activated(const QString&)),this,SLOT(scanTypeChosen(const QString&)));
    connect(ui->comboBox_scantype,SIGNAL(currentIndexChanged(const QString&)),this,SLOT(scanTypeChosen(const QString&)));

    connect(ui->pushButton_start,SIGNAL(clicked()),this,SLOT(startScan()));
    connect(ui->pushButton_stop,SIGNAL(clicked()),this,SLOT(stopScan()));

    connect(ui->pushButton_center,SIGNAL(clicked()),this,SLOT(goHome()));
    connect(ui->pushButton_down,SIGNAL(clicked()),this,SLOT(moveDown()));
    connect(ui->pushButton_right,SIGNAL(clicked()),this,SLOT(moveRight()));
    connect(ui->pushButton_up,SIGNAL(clicked()),this,SLOT(moveUp()));
    connect(ui->pushButton_left,SIGNAL(clicked()),this,SLOT(moveLeft()));
    connect(ui->pushButton_readOffsetFile,SIGNAL(clicked()),this,SLOT(readOffset()));

    connect(ui->pushButton_motor_home,SIGNAL(clicked()),this,SLOT(goMotorHome()));
    connect(ui->pushButton_motor_down,SIGNAL(clicked()),this,SLOT(moveMotorDown()));
    connect(ui->pushButton_motor_up,SIGNAL(clicked()),this,SLOT(moveMotorUp()));
    connect(ui->pushButton_motor_ym,SIGNAL(clicked()),this,SLOT(moveMotorYM()));
    connect(ui->pushButton_motor_yp,SIGNAL(clicked()),this,SLOT(moveMotorYP()));
    connect(ui->pushButton_motor_xp,SIGNAL(clicked()),this,SLOT(moveMotorXP()));
    connect(ui->pushButton_motor_xm,SIGNAL(clicked()),this,SLOT(moveMotorXM()));

    connect(ui->horizontalScrollBar_offsetX,SIGNAL(valueChanged(int)),this,SLOT(updateOffsetViewerX()));
    connect(ui->horizontalScrollBar_offsetY,SIGNAL(valueChanged(int)),this,SLOT(updateOffsetViewerY()));

    connect(ui->pushButton_loadCoeff,SIGNAL(clicked()),this,SLOT(readCoeffTxt()));
    connect(ui->pushButton_saveCoeff,SIGNAL(clicked()),this,SLOT(writeCoeffTxt()));

    connect(ui->lineEdit_datasetname,SIGNAL(editingFinished(void)),this,SLOT(checkPath()));

    connect(ui->checkBox_averageFrames,SIGNAL(clicked(bool)),this,SLOT(slot_updateAverageAngiogram()));

    connect(ui->comboBox_angio,SIGNAL(currentIndexChanged(int)),this,SLOT(slot_updateAngiogramAlgo()));



    connect(ui->checkBox_motorport,SIGNAL(clicked(bool)),this,SLOT(slot_openMotorPort(bool)));
    connect(ui->pushButton_doMosaic,SIGNAL(clicked()),this,SLOT(slot_doMosaic()));
    connect(ui->pushButton_doStack,SIGNAL(clicked()),this,SLOT(slot_doStack()));

    connect(ui->lineEdit_hanningeps,SIGNAL(editingFinished()),this,SLOT(slot_updateHanningThreshold()));
    connect(ui->lineEdit_logeps,SIGNAL(editingFinished()),this,SLOT(slot_updateImageThreshold()));
    connect(ui->checkBox_autoFill,SIGNAL(stateChanged(int)),this,SLOT(autoFillName()));
    this->updateInfo();

    p_save_dir = QDir::home();
    ui->label_directory->setText(p_save_dir.absolutePath());
    connect(ui->pushButton_savedir,SIGNAL(clicked()),this,SLOT(setSaveDir()));

    if (p_settings.contains("default_scans"))
    {
        QStringList empty_list;
        p_saved_scans=p_settings.value("default_scans",empty_list).toStringList();
        for (int i=0;i<p_saved_scans.size();i++)
        {
            ui->comboBox_scanlist->addItem(p_saved_scans.at(i));
        }
    }
    connect(ui->pushButton_add_scanlist,SIGNAL(clicked()),this,SLOT(addDefaultScan()));
    connect(ui->pushButton_load_scanlist,SIGNAL(clicked()),this,SLOT(setDefaultScan()));
    connect(ui->pushButton_clearsettings,SIGNAL(clicked()),this,SLOT(clearCurrentScan()));
    connect(ui->checkBox_invertAxes,SIGNAL(stateChanged(int)),this, SLOT(invertAxes()));
    connect(ui->checkBox_updateLSnumber,SIGNAL(stateChanged(int)),this, SLOT(checkPath()));

    ui->pushButton_start->setEnabled(true);
    ui->pushButton_stop->setEnabled(false);
    ui->lineEdit_datasetname->setEnabled(false);
    connect(ui->pushButton_readOffsetFile,SIGNAL(clicked()),this,SLOT(updateCenterLineEdit()));

    readCoeffTxt();
    updateCenterLineEdit();
    flagMotor=0;

    double* tmp=new double[LINE_ARRAY_SIZE];
    FILE* fp=fopen("C:\\Users\\Public\\Documents\\dispersion_compensation_10x.dat","rb");
    if(fp == 0)
    {
        std::cerr << "No Dispersion Compensation File Found for 10x Objective" << std::endl;
        p_disp_comp_vec_10x=0;
    }
    else
    {
        p_disp_comp_vec_10x=new float[LINE_ARRAY_SIZE];
        fread(tmp,sizeof(double),LINE_ARRAY_SIZE,fp);
        for(int i=0;i<LINE_ARRAY_SIZE;i++) p_disp_comp_vec_10x[i]=(float) tmp[i];
        fclose(fp);
    }
    fp=fopen("C:\\Users\\Public\\Documents\\dispersion_compensation_25x.dat","rb");
    if(fp == 0)
    {
        std::cerr << "No Dispersion Compensation File Found for 25x Objective" << std::endl;
        p_disp_comp_vec_25x=0;
    }
    else
    {
        p_disp_comp_vec_25x=new float[LINE_ARRAY_SIZE];
        fread(tmp,sizeof(double),LINE_ARRAY_SIZE,fp);
        for(int i=0;i<LINE_ARRAY_SIZE;i++) p_disp_comp_vec_25x[i]=(float) tmp[i];
        fclose(fp);
    }
    delete [] tmp;
}

GalvoController::~GalvoController()
{
    slot_openMotorPort(false);
    delete [] p_disp_comp_vec_10x;
    delete [] p_disp_comp_vec_25x;
    delete motors;
    delete ui;
}

void GalvoController::setLineScanPos(int start_x, int start_y, int stop_x, int stop_y)
{
    int nx = ui->lineEdit_nx->text().toInt();
    int ny = ui->lineEdit_ny->text().toInt();
    float width = ui->lineEdit_width->text().toFloat();
    float height = ui->lineEdit_height->text().toFloat();
    if(stop_x > nx) stop_x = nx;
    if(start_x > nx) start_x = nx;

    float ratio_x=width/nx;
    float ratio_y=height/ny;

    p_start_line_x = (1.0*start_x-nx/2.0)*ratio_x;
    p_stop_line_x = (1.0*stop_x-nx/2.0)*ratio_x;
    p_start_line_y = (1.0*start_y-ny/2.0)*ratio_y;
    p_stop_line_y = (1.0*stop_y-ny/2.0)*ratio_y;

    p_line_length=sqrt(pow((p_stop_line_x-p_start_line_x),2)+pow((p_stop_line_y-p_start_line_y),2));

    std::cout<<"----------"<<std::endl;
    std::cout<<"p_line_length: "<<p_line_length<<std::endl;
    std::cout<<"-   -   -"<<std::endl;
    std::cout<<"p_start_line_y: "<<p_start_line_y<<std::endl;
    std::cout<<"p_stop_line_y: "<<p_stop_line_y<<std::endl;
    std::cout<<"p_start_line_x: "<<p_start_line_x<<std::endl;
    std::cout<<"p_stop_line_x: "<<p_stop_line_x<<std::endl;
    std::cout<<"----------"<<std::endl;

    updateInfo();
    return;
}




void GalvoController::setFixedLengthLineScanPos()
{
    float linecenter_x = (p_stop_line_x + p_start_line_x)/2.0;
    float linecenter_y = (p_stop_line_y + p_start_line_y)/2.0;

    QString line_length_cmd_str=ui->lineEdit_width->text();
    float line_length_cmd=line_length_cmd_str.toDouble();

    float ratio_linelength = line_length_cmd/p_line_length;

    float new_start_line_x = (p_start_line_x - linecenter_x) * ratio_linelength + linecenter_x;
    float new_stop_line_x = (p_stop_line_x - linecenter_x) * ratio_linelength + linecenter_x;
    float new_start_line_y = (p_start_line_y - linecenter_y) * ratio_linelength + linecenter_y;
    float new_stop_line_y = (p_stop_line_y - linecenter_y) * ratio_linelength + linecenter_y;

    float new_line_length=sqrt(pow((new_stop_line_x-new_start_line_x),2)+pow((new_stop_line_y-new_start_line_y),2));

    std::cout<<"----------"<<std::endl;
    std::cout<<"p_line_length: "<<p_line_length<<std::endl;
    std::cout<<"new_line_length: "<<new_line_length<<std::endl;
    std::cout<<"line_length_cmd: "<<line_length_cmd<<std::endl;
    std::cout<<"-   -   -"<<std::endl;
    std::cout<<"new_start_line_x: "<<new_start_line_x<<"\t new_start_line_y: "<<new_start_line_y<<std::endl;
    std::cout<<"new_stop_line_x: "<<new_stop_line_x<<"\t new_stop_line_y: "<<new_stop_line_y<<std::endl;
    std::cout<<"p_start_line_x: "<<p_start_line_x<<"\t p_start_line_y: "<<p_start_line_y<<std::endl;
    std::cout<<"p_stop_line_x: "<<p_stop_line_x<<"\t p_stop_line_y: "<<p_stop_line_y<<std::endl;
    std::cout<<"----------"<<std::endl;

    p_start_line_x = new_start_line_x;
    p_stop_line_x = new_stop_line_x;
    p_start_line_y = new_start_line_y;
    p_stop_line_y = new_stop_line_y;
    p_line_length=sqrt(pow((p_stop_line_x-p_start_line_x),2)+pow((p_stop_line_y-p_start_line_y),2));

    return;
}

void GalvoController::setCenterFromLineScan()
{
    p_center_x = (p_stop_line_x + p_start_line_x)/2.0;
    p_center_y = (p_stop_line_y + p_start_line_y)/2.0;

    p_galvos.move(p_center_x,p_center_y);

    std::cout<<"centers: "<<p_center_x<<"/"<<p_center_y<<std::endl;
}

void GalvoController::updateSpeedPiezo()
{
    std::cout<<"piezo speed update"<<std::endl;
    int speedValue = ui->spinBox_piezoSpeed->value();
    motors->getSpeed(speedValue);
}

void GalvoController::jogPiezo()
{
    std::cout<<"piezo go!"<<std::endl;
    motors->PiezoStartJog();
}

void GalvoController::stopPiezo()
{
    std::cout<<"piezo stop!"<<std::endl;
    motors->PiezoStopJog();
}

void GalvoController::turnPiezoOn()
{
    std::cout<<"piezo turning on"<<std::endl;
    motors->PiezoOpenPort();
    flagMotor=1;
}

void GalvoController::turnPiezoOff()
{
    std::cout<<"piezo turning off"<<std::endl;
    motors->PiezoClosePort();
    flagMotor=0;

}

void GalvoController::homePiezo()
{
    std::cout<<"homing piezo"<<std::endl;
    motors->PiezoHome();

}

void GalvoController::updateCenterLineEdit()
{
    ui->lineEdit_center_x->setText(QString::number(p_center_x));
    ui->lineEdit_center_y->setText(QString::number(p_center_y));
}

void GalvoController::invertAxes()
{
    std::cout<<"invert axes!"<<std::endl;
    QString ao_fast = "ao0";
    QString ao_slow = "ao1";
    if(ui->checkBox_invertAxes->checkState())
    {
        p_galvos.setGalvoAxes(ao_slow,ao_fast);

    }
    else
    {
        p_galvos.setGalvoAxes(ao_fast,ao_slow);
    }
}


void GalvoController::slot_openMotorPort(bool flag)
{
    if(flag)
    {
        motors->OpenPort();
        ui->pushButton_motor_down->setEnabled(true);
        ui->pushButton_motor_up->setEnabled(true);
        ui->pushButton_motor_yp->setEnabled(true);
        ui->pushButton_motor_ym->setEnabled(true);
        ui->pushButton_motor_xp->setEnabled(true);
        ui->pushButton_motor_xm->setEnabled(true);
        ui->pushButton_motor_home->setEnabled(true);

    }
    else
    {
        ui->pushButton_motor_down->setEnabled(false);
        ui->pushButton_motor_up->setEnabled(false);
        ui->pushButton_motor_yp->setEnabled(false);
        ui->pushButton_motor_ym->setEnabled(false);
        ui->pushButton_motor_xp->setEnabled(false);
        ui->pushButton_motor_xm->setEnabled(false);
        ui->pushButton_motor_home->setEnabled(false);

        motors->ClosePort();
    }

}

void GalvoController::slot_doMosaic()
{
    // Need to check motors open
    if(ui->checkBox_motorport->isChecked())
    {
        int nx = ui->lineEdit_mosaic_nx->text().toInt();
        int ny = ui->lineEdit_mosaic_ny->text().toInt();
        int overlap_x = ui->lineEdit_mosaic_offset_x->text().toInt();
        int overlap_y = ui->lineEdit_mosaic_offset_y->text().toInt();
        for(int i=0; i<=nx; i++)
        {
            for(int j=0; j<=ny; j++)
            {
                // Move motors
                // Set new datapathname
                // Do acquisition
            }
        }
    }
    else
    {
        QMessageBox msgBox;
        msgBox.setText("Open the motor port prior to doing a mozaic.");
        msgBox.exec();
    }
}

void GalvoController::slot_doStack()
{
    if(ui->checkBox_motorport->isChecked())
    {
        if(p_acq_index == 0)
        {
            p_stack_acquisition = true;
            p_datasetname = ui->lineEdit_datasetname->text();
            // Set new datapathname
            QString stack_name =p_datasetname + QString("_stack_%1").arg(p_acq_index);
            ui->lineEdit_datasetname->setText(stack_name);
            // Do acquisition
            startScan();
        }
        else
        {
            float step_z = ui->lineEdit_stack_step_z->text().toFloat();
            motors->move_dz(-step_z/1000.);
            // Set new datapathname
            QString stack_name =p_datasetname + QString("_stack_%1").arg(p_acq_index);
            ui->lineEdit_datasetname->setText(stack_name);
            // Do acquisition
            startScan();
        }
    }
    else
    {
        QMessageBox msgBox;
        msgBox.setText("Open the motor port prior to doing a z stack.");
        msgBox.exec();
    }
}

void GalvoController::setSaveDir()
{
    dataDir = QFileDialog::getExistingDirectory(this, tr("choose Directory"),
                                                p_save_dir.absolutePath(),
                                                QFileDialog::ShowDirsOnly
                                                | QFileDialog::DontResolveSymlinks);
    p_save_dir.setPath(dataDir);
    ui->label_directory->setText(p_save_dir.absolutePath());
    ui->lineEdit_datasetname->setEnabled(true);
}

void GalvoController::setDefaultScan(void)
{
    QString chosen_scan = ui->comboBox_scanlist->currentText();
    int nx = p_settings.value(chosen_scan+"/nx").toInt();
    int ny = p_settings.value(chosen_scan+"/ny").toInt();
    float width = p_settings.value(chosen_scan+"/width").toFloat();
    float height = p_settings.value(chosen_scan+"/height").toFloat();
    int n_extra = p_settings.value(chosen_scan+"/n_extra").toInt();
    int n_repeat = p_settings.value(chosen_scan+"/n_repeat").toInt();
    float line_rate = p_settings.value(chosen_scan+"/linerate").toFloat();
    float exposure = p_settings.value(chosen_scan+"/exposure").toFloat();
    int scantype = p_settings.value(chosen_scan+"/scantype").toInt();
    int aline_repeat = p_settings.value(chosen_scan+"/aline_repeat",1).toInt();
    bool finite = p_settings.value(chosen_scan+"/finite_acq",true).toBool();
    int n_volumes = p_settings.value(chosen_scan+"/n_volumes",1).toInt();


    ui->lineEdit_nx->setText(QString::number(nx));
    ui->lineEdit_ny->setText(QString::number(ny));
    ui->lineEdit_width->setText(QString::number(width));
    ui->lineEdit_height->setText(QString::number(height));
    ui->lineEdit_extrapoints->setText(QString::number(n_extra));
    ui->lineEdit_fastaxisrepeat->setText(QString::number(n_repeat));
    ui->lineEdit_linerate->setText(QString::number(line_rate));
    ui->lineEdit_exposure->setText(QString::number(exposure));
    ui->lineEdit_aline_repeat->setText(QString::number(aline_repeat));
    ui->comboBox_scantype->setCurrentIndex(scantype);
    ui->lineEdit_nvol->setText(QString::number(n_volumes));
    ui->checkBox_finite_acq->setChecked(finite);
    updateInfo();
}

void GalvoController::addDefaultScan(void)
{
    int nx = ui->lineEdit_nx->text().toInt();
    int ny = ui->lineEdit_ny->text().toInt();
    float width = ui->lineEdit_width->text().toFloat();
    float height = ui->lineEdit_height->text().toFloat();
    int n_repeat = ui->lineEdit_fastaxisrepeat->text().toInt();
    int n_extra = ui->lineEdit_extrapoints->text().toInt();
    float line_rate = ui->lineEdit_linerate->text().toFloat();
    float exposure = ui->lineEdit_exposure->text().toFloat();
    int scantype = ui->comboBox_scantype->currentIndex();
    int aline_repeat = ui->lineEdit_aline_repeat->text().toInt();
    bool finite = ui->checkBox_finite_acq->isChecked();
    int n_volumes = ui->lineEdit_nvol->text().toInt();

    // Add item to combobox
    if (!ui->lineEdit_scanname->text().isEmpty())
    {
        QString new_scan = ui->lineEdit_scanname->text();
        ui->comboBox_scanlist->addItem(new_scan);
        p_settings.setValue(new_scan+"/nx",nx);
        p_settings.setValue(new_scan+"/ny",ny);
        p_settings.setValue(new_scan+"/width",width);
        p_settings.setValue(new_scan+"/height",height);
        p_settings.setValue(new_scan+"/n_extra",n_extra);
        p_settings.setValue(new_scan+"/n_repeat",n_repeat);
        p_settings.setValue(new_scan+"/linerate",line_rate);
        p_settings.setValue(new_scan+"/exposure",exposure);
        p_settings.setValue(new_scan+"/scantype",scantype);
        p_settings.setValue(new_scan+"/aline_repeat",aline_repeat);
        p_settings.setValue(new_scan+"/finite_acq",finite);
        p_settings.setValue(new_scan+"/n_volumes",n_volumes);
        p_saved_scans.append(new_scan);
        p_settings.setValue("default_scans",p_saved_scans);
    }
}

void GalvoController::clearCurrentScan(void)
{
    QString chosen_scan = ui->comboBox_scanlist->currentText();
    p_settings.remove(chosen_scan);
    ui->comboBox_scanlist->removeItem(ui->comboBox_scanlist->currentIndex());
    p_saved_scans.removeDuplicates();
    p_saved_scans.removeAt(p_saved_scans.indexOf(chosen_scan));
}

void GalvoController::updateInfo(void)
{

    QString text("Info:\n");
    int aline_repeat = ui->lineEdit_aline_repeat->text().toInt();
    int nx = ui->lineEdit_nx->text().toInt();
    float width = ui->lineEdit_width->text().toFloat();
    int n_extra = ui->lineEdit_extrapoints->text().toInt();
    float line_rate = ui->lineEdit_linerate->text().toFloat();
    float exposure = ui->lineEdit_exposure->text().toFloat();
    float time_per_pix = 1.0/(line_rate*(nx+n_extra)*aline_repeat)*1e6;
    float speed_x=line_rate*width*(nx+n_extra)/(nx*1000.0);
    float lat_sampling=width/nx;
    float interFrameTime=1/line_rate*1000;
    QString tmp;
    text = text+QString("Current time per pix.:\t%1 us\n").arg(time_per_pix,5,'f',2);

    text=text+QString("Beam speed in x:\t%1 mm/sec\n").arg(speed_x,5,'f',2);

    text=text+QString("Lateral sampling in x:\t%1 um/pix\n").arg(lat_sampling,5,'f',3);

    text=text+QString("Inter B-scan time in x:\t%1 ms\n").arg(interFrameTime,5,'f',3);

    text=text+QString("Linelength:\t\t%1 um\n").arg(p_line_length,5,'f',3);




    if (time_per_pix < 0.9*exposure)
    {
        text=text+QString("WARNING: \nINTEGRATION TIME HIGHER THAN TIME\nPER PIXEL!!!\n");
    }
    ui->label_info->setText(text);
}

void GalvoController::scanTypeChosen(const QString& text)
{
    if (text=="SawTooth")
    {
        ui->label_nx->setText("Nx");
        ui->label_ny->setText("Ny");
        ui->label_width->setText("Width (um)");
        ui->label_height->setEnabled(true);
        ui->lineEdit_height->setEnabled(true);
        ui->checkBox_adjustLength->setChecked(false);
        ui->checkBox_adjustLength->setEnabled(false);
    }
    else if (text=="Triangular")
    {
        ui->label_nx->setText("Nx");
        ui->label_ny->setText("Ny");
        ui->checkBox_adjustLength->setChecked(false);
        ui->checkBox_adjustLength->setEnabled(false);
        ui->label_width->setText("Width (um)");
        ui->label_height->setEnabled(true);
        ui->lineEdit_height->setEnabled(true);
    }
    else if (text=="Line")
    {
        ui->label_nx->setText("N points");
        ui->label_ny->setText("N repeat");
        ui->checkBox_adjustLength->setChecked(true);
        ui->checkBox_adjustLength->setEnabled(true);
        ui->label_width->setText("Length (um)");
        ui->label_height->setEnabled(false);
        ui->lineEdit_height->setEnabled(false);

    }
}

void GalvoController::slot_updateImageThreshold()
{
    float val=ui->lineEdit_logeps->text().toFloat();
    emit sig_updateImageThreshold(val);
}

void GalvoController::slot_updateHanningThreshold()
{
    float val=ui->lineEdit_hanningeps->text().toFloat();
    emit sig_updateHanningThreshold(val);
}

void GalvoController::autoFillName()
{
    bool autoFillFlag=ui->checkBox_autoFill->checkState();
    if(autoFillFlag)
    {
        checkPath();
    }
}

void GalvoController::checkPath()
{
    readOffset();
    bool autoFillFlag=ui->checkBox_autoFill->checkState();
    QString folderName = ui->lineEdit_datasetname->text();
    if(autoFillFlag)
    {
        folderName = ui->comboBox_scanlist->currentText();
    }



    QString newfolderName = folderName;
    QDir pathToData = QDir::cleanPath(dataDir + QDir::separator() + folderName +QDir::separator());
    QString pathToDataTest = pathToData.absolutePath();
    bool folderFlag = true;
    int counter=1;
    bool overwriteFlag=ui->checkBox_overwrite->checkState();
    bool updateLSFlag=ui->checkBox_updateLSnumber->checkState();

    if (!overwriteFlag)
    {
        while(folderFlag)
        {
            if (pathToData.exists())
            {
                QString numberStr;
                numberStr=QString("%1").arg(counter,3);
                newfolderName = folderName+"_"+numberStr;
                pathToData = QDir::cleanPath(dataDir + QDir::separator() + newfolderName +QDir::separator());
                counter++;
            }
            else
            {
                if (updateLSFlag)
                {
                    folderFlag=false;
                    std::cout<<"in line update"<<std::endl;
                    p_line_number_str = readLineNumber();
                    QString linenumber;
                    linenumber=QString("%1").arg(p_line_number_str.toInt(),3);
                    newfolderName = folderName+"_"+linenumber;
                    ui->lineEdit_datasetname->setText(newfolderName);
                }
                else
                {
                    folderFlag=false;
                    ui->lineEdit_datasetname->setText(newfolderName);
                }

            }
        }
    }
}

void GalvoController::automaticCentering()
{
    bool centreAutoFlag=ui->checkBox_centreAuto->checkState();
    bool centreAutoFlagLS=ui->checkBox_centerLS->checkState();

    if(centreAutoFlag)
    {
        readOffset();
    }
    else if(centreAutoFlagLS)
    {
        setCenterFromLineScan();
    }
    else
    {
        goHome();
    }

    updateCenterLineEdit();
}

void GalvoController::startScan()
{
    checkPath();
    automaticCentering();
    qApp->processEvents();

    // Read values
    int nx = ui->lineEdit_nx->text().toInt();
    int ny = ui->lineEdit_ny->text().toInt();
    float width = ui->lineEdit_width->text().toFloat();
    float height = ui->lineEdit_height->text().toFloat();
    int n_repeat = ui->lineEdit_fastaxisrepeat->text().toInt();
    int n_extra = ui->lineEdit_extrapoints->text().toInt();
    float line_rate = ui->lineEdit_linerate->text().toFloat();
    float exposure = ui->lineEdit_exposure->text().toFloat();
    int aline_repeat = ui->lineEdit_aline_repeat->text().toInt();
    // Insuring that the factor is always a multiple of n_repeat to facilitate angios
    int factor = n_repeat;
    if (line_rate/n_repeat > 30) factor = ((int) ((line_rate/n_repeat/30)+1))*n_repeat;
    int telescope = ui->comboBox_telescope->currentIndex();
    double radians_per_volt = 2*3.14159265359/(360*0.8)*(3.0/4.0)/0.95;
    double f1=50.0;
    double f2=100.0;
    bool show_line_flag=ui->checkBox_view_line->isChecked();
    bool finite_acq_flag=ui->checkBox_finite_acq->isChecked();

    // Only go here on first call if we do multiple volumes
    if (finite_acq_flag)
    {
        p_n_volumes = ui->lineEdit_nvol->text().toInt();
        p_finite_acquisition = true;
    }
    switch(telescope)
    {
    case 0:
        f1=54.0;
        f2=300.0;
        break;
    case 1:
        f1=36.0;
        f2=18.0*0.83;
        break;
    case 2:
        f1=50.0;
        f2=100.0;
        break;
    case 3:
        f1=75.0;
        f2=75.0;
        break;
    case 4:
        f1=10.0;
        f2=10.0;
        break;
    }

    int objective = ui->comboBox_objective->currentIndex();
    double fobj=18.0;
    switch(objective)
    {
    case 0:
        fobj=36.0;
        break;
    case 1:
        fobj=18.0;
        break;
    case 2:
        fobj=20.0;
        break;
    case 3:
        fobj=9.0;
        break;
    case 4:
        fobj=7.2;
        break;
    case 5:
        fobj=4.5;
        break;
    case 6:
        fobj=54.0;
        break;
    default:
        break;
    }

    double scale_um_per_volt=(2*fobj*f1/f2*radians_per_volt)*1000.0;
    Converter unit_converter;
    unit_converter.setScale(scale_um_per_volt,scale_um_per_volt);
    p_galvos.setUnitConverter(unit_converter);

    if (ui->checkBox_fringe->isChecked() || ui->checkBox_view_image->isChecked())
    {
        view_timer = new QTimer();
    }
    // Set Camera
    if (p_camera != NULL) delete p_camera;
    unsigned int n_frames_in_one_volume = (ny*n_repeat)*aline_repeat/factor;
    if(n_frames_in_one_volume == 0) n_frames_in_one_volume =1;

#ifndef SIMULATION
    p_camera=new Camera((nx+n_extra)*factor,exposure,n_frames_in_one_volume);
#else
    p_camera=new SoftwareCamera((nx+n_extra)*factor,exposure,n_frames_in_one_volume);
#endif
    if(p_finite_acquisition || p_stack_acquisition)
    {
        connect(p_camera,SIGNAL(volume_done()),this,SLOT(stopFiniteScan()));
    }

    // If we are saving, setup for it
    if (show_line_flag)
    {
        std::cout<<"show flag!"<<std::endl;
        p_start_viewline=ui->lineEdit_startLine->text().toInt();
        p_stop_viewline=ui->lineEdit_stopLine->text().toInt();
    }

    if (ui->checkBox_adjustLength->isChecked())
    {
        std::cout<<"Setting fixed length..."<<std::endl;
        setFixedLengthLineScanPos();
        std::cout<<"...done!"<<std::endl;
    }
    updateInfo();
    qApp->processEvents();

    if (ui->checkBox_save->isChecked())
    {
        p_block_size = (int) ((512.0*256.0)/nx/factor);

        p_data_saver = new DataSaver((nx+n_extra)*factor,p_block_size);
        p_data_saver->setDatasetName(ui->lineEdit_datasetname->text());
        p_data_saver->setDatasetPath(p_save_dir.absolutePath());

        // AI
        if(p_ai != 0) delete p_ai;

        p_ai=new AnalogInput(AIAOSAMPRATE);

        p_ai_data_saver = new Float64DataSaver(N_AI_CHANNELS,AIAOSAMPRATE,256,"ai");
        p_ai_data_saver->setDatasetName(ui->lineEdit_datasetname->text());
        p_ai_data_saver->setDatasetPath(p_save_dir.absolutePath());



        //QString info;
        QString info = QString("nx: %1\n").arg(nx);
        info=info+QString("ny: %1\n").arg(ny);
        info=info+QString("n_repeat: %1\n").arg(n_repeat);
        if (ui->comboBox_scantype->currentText() == "Line")
        {
            info=info+QString("width: %1\n").arg(p_line_length);
        }
        else
        {
            info=info+QString("width: %1\n").arg(width);
        }
        info=info+QString("height: %1\n").arg(height);
        info=info+QString("n_extra: %1\n").arg(n_extra);
        info=info+QString("line_rate: %1\n").arg(line_rate);
        info=info+QString("exposure: %1\n").arg(exposure);
        info=info+QString("alinerepeat: %1\n").arg(aline_repeat);
        info=info+"scantype: "+ui->comboBox_scantype->currentText()+"\n";
        info=info+QString("center_x: %1\n").arg(p_center_x);
        info=info+QString("center_y: %1\n").arg(p_center_y);
        info=info+QString("offset_x: %1\n").arg(p_offset_x);
        info=info+QString("offset_y: %1\n").arg(p_offset_y);
        info=info+QString("coeff_x: %1\n").arg(p_coeff_x);
        info=info+QString("coeff_y: %1\n").arg(p_coeff_y);

        QString objective = ui->comboBox_objective->currentText();
        info=info+QString("objective: %1\n").arg(objective.toUtf8().constData());

        if (ui->checkBox_adjustLength->isChecked())
        {
            info=info+QString("start_line_x: %1\n").arg(p_start_line_x);
            info=info+QString("stop_line_x: %1\n").arg(p_stop_line_x);
            info=info+QString("start_line_y: %1\n").arg(p_start_line_y);
            info=info+QString("stop_line_y: %1\n").arg(p_stop_line_y);
            info=info+QString("line_length: %1\n").arg(p_line_length);
        }


        p_data_saver->addInfo(info);
        connect(p_data_saver,SIGNAL(available(int)),ui->lcdNumber_saveqsize,SLOT(display(int)));
        connect(p_data_saver,SIGNAL(filenumber(int)),this,SLOT(displayFileNumber(int)));
    }
    p_camera->Open();
    p_camera->SetCameraString("FPA Sensitivity",ui->comboBox_sensitivity->currentText().toUtf8().constData());
    p_camera->SetCameraString("Operational Setting",ui->comboBox_opr->currentText().toUtf8().constData());
    p_camera->SetCameraString("Exposure Modes","Exp at Max Rate");
    if (exposure < 8.45f) exposure = 8.45f;
    if (exposure > 100.0f) exposure = 100.0f;
    p_camera->SetCameraNumeric("Exposure Time",exposure);

    p_camera->ConfigureForSingleGrab();
    if(ui->checkBox_fringe->isChecked())
    {
        p_fringe_view = new FringeViewer(0,nx+n_extra);
        connect(view_timer,SIGNAL(timeout()),p_fringe_view,SLOT(updateView()));
        p_fringe_view->show();
        if(ui->checkBox_placeImage->isChecked())
            p_fringe_view->move(75,150);
        p_camera->setFringeViewer(p_fringe_view);
    }
    if(ui->checkBox_view_image->isChecked())
    {
        int view_depth = ui->lineEdit_displayPoints->text().toInt();
        float hpf_time_constant = ui->lineEdit_doppler_hpf->text().toFloat()/1000.;
        float spatial_kernel_size=ui->lineEdit_doppler_spatial_kernel->text().toFloat();
        float line_period = 1.0f/line_rate/(nx+n_extra);
        float dimx = width/nx;
        float dimz = 3.5;
        p_image_view = new ImageViewer(0,nx+n_extra,n_extra,ny, view_depth,n_repeat, hpf_time_constant,line_period,spatial_kernel_size,dimz,dimx,factor);
        p_image_view->updateHanningThreshold(ui->lineEdit_hanningeps->text().toFloat());
        p_image_view->updateImageThreshold(ui->lineEdit_logeps->text().toFloat());
        p_image_view->updateAngioAlgo(ui->comboBox_angio->currentIndex());
        p_image_view->checkLine(show_line_flag,p_start_viewline,p_stop_viewline);
        int dispersion = ui->comboBox_dispersion->currentIndex();
        if(dispersion == 2 && p_disp_comp_vec_10x != 0) p_image_view->set_disp_comp_vect(p_disp_comp_vec_10x);
        if(dispersion == 4 && p_disp_comp_vec_25x != 0) p_image_view->set_disp_comp_vect(p_disp_comp_vec_25x);
        bool averageAngioFlag=ui->checkBox_averageFrames->isChecked();
        p_image_view->updateAngioAverageFlag(averageAngioFlag);
        connect(view_timer,SIGNAL(timeout()),p_image_view,SLOT(updateView()));
        connect(this,SIGNAL(sig_updateAveragingFlag(bool)),p_image_view,SLOT(updateAngioAverageFlag(bool)));
        connect(this,SIGNAL(sig_updateAveragingAlgo(int)),p_image_view,SLOT(updateAngioAlgo(int)));



        connect(this,SIGNAL(sig_updateHanningThreshold(float)),p_image_view,SLOT(updateHanningThreshold(float)));
        connect(this,SIGNAL(sig_updateImageThreshold(float)),p_image_view,SLOT(updateImageThreshold(float)));
        connect(this,SIGNAL(sig_updateViewLinePositions(bool,int,int)),p_image_view,SLOT(updateViewLinePositions(bool,int,int)));

        connect(p_image_view,SIGNAL(sig_updateLineScanPos(int,int,int,int)),this,SLOT(setLineScanPos(int,int,int,int)));
        if(ui->checkBox_placeImage->isChecked())
            p_image_view->move(200,150);
        p_image_view->show();
        p_camera->setImageViewer(p_image_view);
        QString folderName = ui->comboBox_scanlist->currentText();
        if (folderName=="RBCpassage_1_5_ms")
        {
            p_image_view->setCurrentViewModeStruct();
        }
    }
    if (ui->checkBox_save->isChecked())
    {
        p_camera->setDataSaver(p_data_saver);
        p_data_saver->writeInfoFile();
        p_data_saver->startSaving();
        p_ai->SetDataSaver(p_ai_data_saver);
        p_ai_data_saver->startSaving();
        p_ai->Start();

    }

    p_camera->Start();

    if (ui->checkBox_fringe->isChecked() || ui->checkBox_view_image->isChecked())
    {
        view_timer->start(30);
    }

    // Set ramp
    if (ui->comboBox_scantype->currentText() == "SawTooth")
    {
        p_galvos.setSawToothRamp(-width/2,-height/2,+width/2,height/2,nx,ny,n_extra,n_repeat,line_rate,aline_repeat);
        p_galvos.move(p_center_x, p_center_y);

    }
    else if (ui->comboBox_scantype->currentText() == "Triangular")
    {
        p_galvos.setTriangularRamp(p_center_x-width/2,p_center_y-height/2,p_center_x+width/2,p_center_y+height/2,nx,ny,n_extra,line_rate);
    }
    else if(ui->comboBox_scantype->currentText() == "Line")
    {
        p_galvos.setLineRamp(p_center_x+p_start_line_x,p_center_y+p_start_line_y,
                             p_center_x+p_stop_line_x,p_center_y+p_stop_line_y,nx,ny,n_extra,line_rate);
    }
    p_galvos.setTrigDelay(ui->lineEdit_shift->text().toFloat());
    // Start generating

    p_galvos.startTask();
    ui->pushButton_start->setEnabled(false);
    ui->pushButton_stop->setEnabled(true);
}

void GalvoController::displayFileNumber(int block_number)
{
    int ny = ui->lineEdit_ny->text().toInt();
    int n_repeat = ui->lineEdit_fastaxisrepeat->text().toInt();

    ui->lcdNumber_acquisition->display(p_block_size*block_number/(ny*n_repeat));
    return;
}

void GalvoController::stopFiniteScan()
{
    if(p_stack_acquisition)
    {
        if(p_finite_acquisition)
        {
            p_n_volumes--;
            if(p_n_volumes == 0) stopScan();
        }
        else
        {
            stopScan();
        }
        return;
    }
    else
    {
        if(p_finite_acquisition)
        {
            p_n_volumes--;
            if(p_n_volumes == 0) stopScan();
        }
    }
}

void GalvoController::stopScan()
{
    // Saver stop
    // Needs to be stopped first due to potential deadlock, will
    // stop when next block size if filled.

    if(p_data_saver)
    {
        bool show_line_flag=ui->checkBox_view_line->isChecked();
        if (show_line_flag)
        {
            QString info;
            QString tmp=QString("start_line: %1\n").arg(p_start_viewline);
            info=info+QString("stop_line: %1\n").arg(p_stop_viewline);
            p_data_saver->addInfo(info);
            p_data_saver->writeInfoFile();
        }



        p_data_saver->stopSaving();
        p_ai_data_saver->stopSaving();
    }

    // camera stop
    // Slight danger of locking if buffer was full and camera is still putting fast
    // Need to have a large acquire at end of thread maybe?
    p_camera->Stop();
    // Deleting saver after camera stop because there will be some calls to put...

    if(p_data_saver)
    {
        delete p_data_saver;
        p_data_saver = 0;
        p_ai->Stop();
        delete p_ai;
        p_ai=0;
        delete p_ai_data_saver;
        p_ai_data_saver=0;
    }

    view_timer->stop();


    if(p_fringe_view)
    {
        p_fringe_view->Close();
        delete p_fringe_view;
        p_fringe_view = 0;
    }

    if(p_image_view)
    {
        p_image_view->close();

        delete p_image_view;
        p_image_view = 0;
    }

    // Stop galvos, close camera
    p_galvos.stopTask();
    p_camera->Close();
    if(!p_stack_acquisition)
    {
        ui->pushButton_start->setEnabled(true);
        ui->pushButton_stop->setEnabled(false);
    }
    else
    {
        p_acq_index ++;
        if( p_acq_index < ui->lineEdit_stack_nz->text().toInt())
        {
            QThread::sleep(2);
            slot_doStack();
        }
        else
        {
            p_acq_index = 0;
            p_stack_acquisition = false;
            ui->pushButton_start->setEnabled(true);
            ui->pushButton_stop->setEnabled(false);
        }
    }
}

void GalvoController::moveUp(void)
{
    p_center_y-=1;
    p_galvos.move(p_center_x,p_center_y);
    updateCenterLineEdit();
}

void GalvoController::moveDown(void)
{
    p_center_y+=1;
    p_galvos.move(p_center_x,p_center_y);
    updateCenterLineEdit();
}

void GalvoController::moveRight(void)
{
    p_center_x-=1;
    p_galvos.move(p_center_x,p_center_y);
    updateCenterLineEdit();
}

void GalvoController::moveLeft(void)
{
    p_center_x+=1;
    p_galvos.move(p_center_x,p_center_y);
    updateCenterLineEdit();
}

void GalvoController::goHome(void)
{
    p_center_x=0;
    p_center_y=0;
    p_galvos.move(p_center_x,p_center_y);
    std::cout << "p_center_x: " << p_center_x << "/ p_center_y: " << p_center_y << std::endl;
    updateCenterLineEdit();
}

void GalvoController::moveMotorUp(void)
{
    float step_um = ui->lineEdit_motor_step_size->text().toFloat();
    motors->move_dz(step_um/1000);
}

void GalvoController::moveMotorDown(void)
{
    float step_um = ui->lineEdit_motor_step_size->text().toFloat();
    motors->move_dz(-step_um/1000);
}

void GalvoController::moveMotorYP(void)
{
    float step_um = ui->lineEdit_motor_step_size->text().toFloat();
    motors->move_dy(step_um/1000);
}

void GalvoController::moveMotorYM(void)
{
    float step_um = ui->lineEdit_motor_step_size->text().toFloat();
    motors->move_dy(-step_um/1000);
}

void GalvoController::moveMotorXP(void)
{
    float step_um = ui->lineEdit_motor_step_size->text().toFloat();
    motors->move_dx(step_um/1000);
}

void GalvoController::moveMotorXM(void)
{
    float step_um = ui->lineEdit_motor_step_size->text().toFloat();
    motors->move_dx(-step_um/1000);

}

void GalvoController::goMotorHome(void)
{
    motors->Home();
}

void GalvoController::writeCoeffTxt(void)
{
    std::cout<<"in readcoeffTxt"<<std::endl;
    QFile file("C:/git-projects/micro-oct-acq/userCoefficients.txt");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return;
    QTextStream out(&file);
    p_offset_x_added=readOffsetX();
    p_offset_y_added=readOffsetY();
    out << p_offset_x_added << "/" << p_offset_y_added << "\n";
    file.close();
}

void GalvoController::readCoeffTxt(void)
{
    QFile file("C:/git-projects/micro-oct-acq/coefficientsOCT.txt");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;
    QTextStream in(&file);
    QString line = in.readLine();
    file.close();
    QStringList fields = line.split("/");
    p_offset_x=fields.at(6).toFloat();
    p_coeff_x=fields.at(7).toFloat();
    p_coeff_x_quad=fields.at(8).toFloat();
    p_coeff_xy=fields.at(9).toFloat();
    p_coeff_xy_quad=fields.at(10).toFloat();
    p_coeff_xxy=fields.at(11).toFloat();

    p_offset_y=fields.at(0).toFloat();
    p_coeff_yx=fields.at(1).toFloat();
    p_coeff_yx_quad=fields.at(2).toFloat();
    p_coeff_y=fields.at(3).toFloat();
    p_coeff_y_quad=fields.at(4).toFloat();
    p_coeff_yyx=fields.at(5).toFloat();

    QFile fileCoeff("C:\git-projects\micro-oct-acq-angiolive/userCoefficients.txt");
    if (!fileCoeff.open(QIODevice::ReadOnly | QIODevice::Text))
        return;
    QTextStream inCoeff(&fileCoeff);
    QString lineCoeff = inCoeff.readLine();
    fileCoeff.close();
    QStringList fieldsCoeff = lineCoeff.split("/");

    p_offset_x_added=fieldsCoeff.at(0).toFloat();
    p_offset_y_added=fieldsCoeff.at(1).toFloat();

    ui->horizontalScrollBar_offsetX->setValue(p_offset_x_added);
    ui->horizontalScrollBar_offsetY->setValue(p_offset_y_added);
    ui->lineEdit_offsetX->setText(QString::number(p_offset_x_added));
    ui->lineEdit_offsetY->setText(QString::number(p_offset_y_added));

    updateOffsetViewerX();
    updateOffsetViewerY();
}

void GalvoController::readOffset(void)
{
    p_galvos.move(0,0);

    QFile file("C:/git-projects/multiphoton/coordinates.txt");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;
    QTextStream in(&file);
    QString line = in.readLine();
    file.close();
    QStringList fields = line.split("/");
    p_line_number=fields.at(2).toInt();
    p_center_y=fields.at(1).toFloat();
    p_center_x=fields.at(0).toFloat();

    ui->lineEdit_readOffsetX->setText(QString::number(p_center_x));
    ui->lineEdit_readOffsetY->setText(QString::number(p_center_y));

    ui->lineEdit_readLineNumber->setText(QString::number(p_line_number));

    float p_center_y_mod=(p_offset_x+p_offset_x_added)+p_center_x*p_coeff_x+p_center_x*p_coeff_x_quad+p_center_y*p_coeff_xy+p_center_y*p_coeff_xy_quad+p_coeff_xxy*p_center_y*p_center_x;
    float p_center_x_mod=(p_offset_y+p_offset_y_added)+p_center_x*p_coeff_yx+p_center_x*p_coeff_yx_quad+p_coeff_y*p_center_y+p_center_y*p_coeff_y_quad+p_coeff_yyx*p_center_y*p_center_x;

    p_center_y=p_center_x_mod;
    p_center_x=-p_center_y_mod;

    p_galvos.move(p_center_x,p_center_y);


}

void GalvoController::updateOffset(void)
{
    p_galvos.move(0,0);
    QFile file("C:/git-projects/multiphoton/coordinates.txt");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return;
    QTextStream in(&file);
    QString line = in.readLine();
    file.close();
    QStringList fields = line.split("/");
    p_line_number=fields.at(2).toInt();
    p_center_x=fields.at(0).toFloat();
    p_center_y=fields.at(1).toFloat();

    float p_center_y_mod=(p_offset_x+p_offset_x_added)+p_center_x*p_coeff_x+p_center_x*p_coeff_x_quad+p_center_y*p_coeff_xy+p_center_y*p_coeff_xy_quad+p_coeff_xxy*p_center_y*p_center_x;
    float p_center_x_mod=(p_offset_y+p_offset_y_added)+p_center_x*p_coeff_yx+p_center_x*p_coeff_yx_quad+p_coeff_y*p_center_y+p_center_y*p_coeff_y_quad+p_coeff_yyx*p_center_y*p_center_x;

    p_center_y=p_center_x_mod;
    p_center_x=-p_center_y_mod;

    p_galvos.move(p_center_x,p_center_y);
}

void GalvoController::updateOffsetViewerX(void)
{
    std::cout<<"in update: previous value: offsetX:"<< p_offset_x+p_offset_x_added <<std::endl;
    p_offset_x_added=readOffsetX();
    ui->lineEdit_offsetX->setText(QString::number(p_offset_x_added));
    updateOffset();
    std::cout<<"in update: offsetY:"<< p_offset_y+p_offset_y_added <<std::endl;
    std::cout<<"in update: offsetX:"<< p_offset_x+p_offset_x_added <<std::endl;
}

void GalvoController::updateOffsetViewerY(void)
{
    std::cout<<"in update: previous value: offsetY:"<< p_offset_y+p_offset_y_added <<std::endl;
    p_offset_y_added=readOffsetY();
    ui->lineEdit_offsetY->setText(QString::number(p_offset_y_added));
    updateOffset();
    std::cout<<"in update: offsetY:"<< p_offset_y+p_offset_y_added <<std::endl;
    std::cout<<"in update: offsetX:"<< p_offset_x+p_offset_x_added <<std::endl;

}

float GalvoController::readOffsetX(void)
{
    p_offset_x_added=ui->horizontalScrollBar_offsetX->value();
    return p_offset_x_added;
}

float GalvoController::readOffsetY(void)
{
    p_offset_y_added=ui->horizontalScrollBar_offsetY->value();
    return p_offset_y_added;
}

float GalvoController::readOffsetX_2(void)
{
    p_offset_x_added_2=ui->horizontalScrollBar_offsetX_2->value();
    return p_offset_x_added_2;
}

float GalvoController::readOffsetY_2(void)
{
    p_offset_y_added_2=ui->horizontalScrollBar_offsetY_2->value();
    return p_offset_y_added_2;
}

QString GalvoController::readLineNumber(void)
{
    p_line_number_str=ui->lineEdit_readLineNumber->text();
    return p_line_number_str;
}

void GalvoController::slot_updateViewLinePositions(void)
{
    p_start_viewline = ui->lineEdit_startLine->text().toInt();
    p_stop_viewline = ui->lineEdit_stopLine->text().toInt();
    bool show_line_flag=ui->checkBox_view_line->isChecked();
    emit sig_updateViewLinePositions(show_line_flag,p_start_viewline,p_stop_viewline);
}

void GalvoController::slot_updateAverageAngiogram(void)
{
    bool averageAngioFlag=ui->checkBox_averageFrames->isChecked();
    emit sig_updateAveragingFlag(averageAngioFlag);
}

void GalvoController::slot_updateAngiogramAlgo(void)
{
    int angioAlgo=ui->comboBox_angio->currentIndex();
    emit sig_updateAveragingAlgo(angioAlgo);
}
