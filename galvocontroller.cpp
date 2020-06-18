#include "galvocontroller.h"
#include <iostream>
#include <QIntValidator>
#include <QDoubleValidator>
#include <QDir>
#include <QFileDialog>
#include <QTimer>
#include <QStringList>
#include <QTextStream>
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

    motors = new MotorClass();

    double radians_per_volt = 2*3.14159265359/(360*0.8);
    double f1=50.0;
    double f2=100.0;
    double fobj=18.0;
    double scale_um_per_volt=(2*fobj*f1/f2*radians_per_volt)*1000.0;
    Converter unit_converter;
    unit_converter.setScale(scale_um_per_volt,scale_um_per_volt);
    p_galvos.setUnitConverter(unit_converter);

    QIntValidator* validator=new QIntValidator(1,3000,this);
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



    connect(ui->comboBox_scantype,SIGNAL(activated(const QString&)),this,SLOT(scanTypeChosen(const QString&)));

    connect(ui->pushButton_start,SIGNAL(clicked()),this,SLOT(startScan()));
    connect(ui->pushButton_stop,SIGNAL(clicked()),this,SLOT(stopScan()));

    connect(ui->pushButton_center,SIGNAL(clicked()),this,SLOT(goHome()));
    connect(ui->pushButton_down,SIGNAL(clicked()),this,SLOT(moveDown()));
    connect(ui->pushButton_right,SIGNAL(clicked()),this,SLOT(moveRight()));
    connect(ui->pushButton_up,SIGNAL(clicked()),this,SLOT(moveUp()));
    connect(ui->pushButton_left,SIGNAL(clicked()),this,SLOT(moveLeft()));
    connect(ui->pushButton_readOffsetFile,SIGNAL(clicked()),this,SLOT(readOffset()));
    //connect(ui->horizontalScrollBar_offsetX,SIGNAL(valueChanged(int)),this,SLOT(updateOffset()));
    //connect(ui->horizontalScrollBar_offsetY,SIGNAL(valueChanged(int)),this,SLOT(updateOffset()));
    connect(ui->horizontalScrollBar_offsetX,SIGNAL(valueChanged(int)),this,SLOT(updateOffsetViewerX()));
    connect(ui->horizontalScrollBar_offsetY,SIGNAL(valueChanged(int)),this,SLOT(updateOffsetViewerY()));
    //connect(ui->horizontalScrollBar_coeffY,SIGNAL(valueChanged(int)),this,SLOT(updateOffset()));
    //connect(ui->horizontalScrollBar_coeffX,SIGNAL(valueChanged(int)),this,SLOT(updateOffset()));
    connect(ui->pushButton_loadCoeff,SIGNAL(clicked()),this,SLOT(readCoeffTxt()));
    connect(ui->pushButton_saveCoeff,SIGNAL(clicked()),this,SLOT(writeCoeffTxt()));

    connect(ui->lineEdit_datasetname,SIGNAL(editingFinished(void)),this,SLOT(checkPath()));



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
    //updateOffsetViewerX();
    //updateOffsetViewerY();
    //updateCoeffViewerX();
    //updateCoeffViewerY();
    updateCenterLineEdit();
    flagMotor=0;
}

GalvoController::~GalvoController()
{
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

    p_start_line_x = (1.0*start_x-nx/2.0)*width/nx;
    p_stop_line_x = (1.0*stop_x-nx/2.0)*width/nx;
    p_start_line_y = (1.0*start_y-ny/2.0)*height/ny;
    p_stop_line_y = (1.0*stop_y-ny/2.0)*height/ny;
    //std::cerr << p_start_line_x << " " << p_start_line_y << " " << p_stop_line_x << " " << p_stop_line_y << std::endl;
    return;
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
    }
    else
    {
        motors->ClosePort();
    }

}

void GalvoController::slot_doMosaic()
{
    // Need to check motors open
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

void GalvoController::slot_doStack()
{
    // Need to check motors open
    // Need to check motors open
    int nz = ui->lineEdit_stack_nz->text().toInt();
    int step_z = ui->lineEdit_stack_step_z->text().toFloat();
    int start_z = ui->lineEdit_stack_start_offset_z->text().toInt();
    for(int i=0; i<=nz; i++)
    {
        // Move motors
        motors->move_az(step_z);
            // Set new datapathname
            // Do acquisition
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
    tmp.sprintf("Current time per pix.:\t%5.2f us\n",time_per_pix);
    text = text+tmp;
    tmp.sprintf("Current exposure:\t%5.2f us\n",exposure);
    text=text+tmp;
    tmp.sprintf("Beam speed in x:\t%5.2f mm/sec\n",speed_x);
    text=text+tmp;
    tmp.sprintf("Lateral sampling in x:\t%5.3f um/pix\n",lat_sampling);
    text=text+tmp;
    tmp.sprintf("Inter B-scan time in x:\t%5.3f ms\n",interFrameTime);
    text=text+tmp;

    if (time_per_pix < 0.9*exposure)
    {
        tmp.sprintf("WARNING: \nINTEGRATION TIME HIGHER THAN TIME\nPER PIXEL!!!\n");
        text=text+tmp;
    }
    ui->label_info->setText(text);
}

void GalvoController::scanTypeChosen(const QString& text)
{
    if (text=="Sawtooth")
    {
        ui->label_nx->setText("Nx");
        ui->label_ny->setText("Ny");
    }
    else if (text=="Triangular")
    {
        ui->label_nx->setText("Nx");
        ui->label_ny->setText("Ny");
    }
    else if (text=="Line")
    {
        ui->label_nx->setText("N points");
        ui->label_ny->setText("N repeat");
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
    bool updateLSnumberFlag=ui->checkBox_updateLSnumber->checkState();
    bool RBCflag=((newfolderName=="RBCpassage_1_5_ms")||(newfolderName=="CapVel"));
    bool updateLSFlag=(updateLSnumberFlag && RBCflag);
    std::cout<<"LS flag: "<<updateLSnumberFlag<< "RBC flag: "<< RBCflag<<" flag: "<< RBCflag<<std::endl;

    if (!overwriteFlag)
    {
        while(folderFlag)
        {
            if (pathToData.exists())
            {
                QString numberStr;
                numberStr.sprintf("%03d", counter);
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
                    newfolderName = folderName+"_"+p_line_number_str;
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
    if(centreAutoFlag)
    {
        QString folderName = ui->comboBox_scanlist->currentText();
        if (folderName=="RBCpassage_1_5_ms")
        {
            std::cout<<"RBC passage: aligning on 2P linescan"<<std::endl;
            readOffset();
        }
        else if(folderName=="angio_40reps_150Hz")
        {
            goHome();
            std::cout<<"40 reps: going Home centered with 2P"<<std::endl;
            p_center_x=readOffsetX();
            p_center_y=readOffsetY();
            p_galvos.move(p_center_x,p_center_y);
        }
        else
        {
            std::cout<<"Other mode selected: aligning on 2P linescan"<<std::endl;
            readOffset();
        }
    }
    else
    {
        std::cout<<"Going home:"<<std::endl;
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
    int p_start_line=0;
    int p_stop_line=0;

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
#ifndef SIMULATION
    p_camera=new Camera((nx+n_extra)*factor,exposure);
#else
    p_camera=new SoftwareCamera((nx+n_extra)*factor,exposure);
#endif

    // If we are saving, setup for it
    if (show_line_flag)
    {
        std::cout<<"show flag!"<<std::endl;
        p_start_line=ui->lineEdit_startLine->text().toInt();
        p_stop_line=ui->lineEdit_stopLine->text().toInt();
    }


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


        QString info;
        QString tmp;
        info.sprintf("nx: %d\n",nx);
        info=info+tmp.sprintf("ny: %d\n",ny);
        info=info+tmp.sprintf("n_repeat: %d\n",n_repeat);
        info=info+tmp.sprintf("width: %f\n",width);
        info=info+tmp.sprintf("height: %f\n",height);
        info=info+tmp.sprintf("n_extra: %d\n",n_extra);
        info=info+tmp.sprintf("line_rate: %f\n",line_rate);
        info=info+tmp.sprintf("exposure: %f\n",exposure);
        info=info+tmp.sprintf("alinerepeat: %d\n",aline_repeat);
        info=info+"scantype: "+ui->comboBox_scantype->currentText()+"\n";
        info=info+tmp.sprintf("center_x: %f\n",p_center_x);
        info=info+tmp.sprintf("center_y: %f\n",p_center_y);
        info=info+tmp.sprintf("offset_x: %f\n",p_offset_x);
        info=info+tmp.sprintf("offset_y: %f\n",p_offset_y);
        info=info+tmp.sprintf("coeff_x: %f\n",p_coeff_x);
        info=info+tmp.sprintf("coeff_y: %f\n",p_coeff_y);

        if (show_line_flag)
        {
            std::cout<<"show flag!  - in save"<<std::endl;
            info=info+tmp.sprintf("start_line: %d\n",p_start_line);
            info=info+tmp.sprintf("stop_line: %d\n",p_stop_line);
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
        p_image_view->checkLine(show_line_flag,p_start_line,p_stop_line);
        connect(view_timer,SIGNAL(timeout()),p_image_view,SLOT(updateView()));
        connect(this,SIGNAL(sig_updateHanningThreshold(float)),p_image_view,SLOT(updateHanningThreshold(float)));
        connect(this,SIGNAL(sig_updateImageThreshold(float)),p_image_view,SLOT(updateImageThreshold(float)));
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
    std::cout<<"flagMotor:"<<std::endl;

    std::cout<<"flagMotor:"<<flagMotor<<std::endl;


    if (ui->checkBox_speckle_mod->isChecked() && flagMotor)
    {
        motors->PiezoStartJog();
        std::cout<<"starting piezo"<<std::endl;
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

void GalvoController::stopScan()
{
    // Saver stop
    // Needs to be stopped first due to potential deadlock, will
    // stop when next block size if filled.



    if (ui->checkBox_speckle_mod->isChecked() && flagMotor)
    {
        motors->PiezoStopJog();
        std::cout<<"stopping piezo"<<std::endl;
    }

    if(p_data_saver)
    {
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
    ui->pushButton_start->setEnabled(true);
    ui->pushButton_stop->setEnabled(false);
}

void GalvoController::moveUp(void)
{
    p_center_y-=10;
    p_galvos.move(p_center_x,p_center_y);
}

void GalvoController::moveDown(void)
{
    p_center_y+=10;
    p_galvos.move(p_center_x,p_center_y);
    updateCenterLineEdit();
}

void GalvoController::moveRight(void)
{
    p_center_x-=10;
    p_galvos.move(p_center_x,p_center_y);
    updateCenterLineEdit();
}

void GalvoController::moveLeft(void)
{
    p_center_x+=10;
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
    std::cout<<"in readcoeffTxt"<<std::endl;
    QFile file("C:/git-projects/micro-oct-acq/coefficientsOCT.txt");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return;
    QTextStream in(&file);
    QString line = in.readLine();
    file.close();
    QStringList fields = line.split("/");

    std::cout<<"in readcoeffTxt"<<std::endl;


    p_offset_x=fields.at(6).toFloat();
    p_coeff_x=fields.at(7).toFloat();
    p_coeff_x_quad=fields.at(8).toFloat();
    p_coeff_xy=fields.at(9).toFloat();
    p_coeff_xy_quad=fields.at(10).toFloat();
    std::cout<<"in readcoeffTxt"<<std::endl;
    p_coeff_xxy=fields.at(11).toFloat();

    p_offset_y=fields.at(0).toFloat();
    p_coeff_yx=fields.at(1).toFloat();
    p_coeff_yx_quad=fields.at(2).toFloat();
    p_coeff_y=fields.at(3).toFloat();
    std::cout<<"in readcoeffTxt"<<std::endl;

    p_coeff_y_quad=fields.at(4).toFloat();
    p_coeff_yyx=fields.at(5).toFloat();

    std::cout<<"in readcoeffTxt"<<std::endl;
    QFile fileCoeff("C:/git-projects/micro-oct-acq/userCoefficients.txt");
    if (!fileCoeff.open(QIODevice::ReadOnly | QIODevice::Text))
            return;
    QTextStream inCoeff(&fileCoeff);
    QString lineCoeff = inCoeff.readLine();
    fileCoeff.close();
    QStringList fieldsCoeff = lineCoeff.split("/");

    std::cout<<"in readcoeffTxt 2"<<std::endl;

    p_offset_x_added=fieldsCoeff.at(0).toFloat();
    p_offset_y_added=fieldsCoeff.at(1).toFloat();

    ui->horizontalScrollBar_offsetX->setValue(p_offset_x_added);
    ui->horizontalScrollBar_offsetY->setValue(p_offset_y_added);
    ui->lineEdit_offsetX->setText(QString::number(p_offset_x_added));
    ui->lineEdit_offsetY->setText(QString::number(p_offset_y_added));

    updateOffsetViewerX();
    updateOffsetViewerY();
    std::cout<<"readcoeffTxt done!"<<std::endl;
}

void GalvoController::readOffset(void)
{
    p_galvos.move(0,0);
    // Could be changed to f
    //QString fileName = QFileDialog::getOpenFileName(this,
        //tr("Open Galvo Offset File"), QDir::homePath() , tr("Text Files (*.txt)"));

    //std::cerr << fileName.toUtf8().constData() << std::endl;
    //QFile file(fileName);
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
    std::cout << "1a. p_center_x: " << p_center_x << "/ p_center_y: " << p_center_y << std::endl;

    ui->lineEdit_readLineNumber->setText(QString::number(p_line_number));
    std::cout << "1b. p_offset_y_added: " << p_offset_x_added << "/ p_offset_y_added: " << p_offset_y_added << std::endl;
    std::cout << "1c. p_offset_y: " << p_offset_y << "/ p_offset_x: " << p_offset_x << std::endl;

    float p_center_y_mod=(p_offset_x+p_offset_x_added)+p_center_x*p_coeff_x+p_center_x*p_coeff_x_quad+p_center_y*p_coeff_xy+p_center_y*p_coeff_xy_quad+p_coeff_xxy*p_center_y*p_center_x;

    float p_center_x_mod=(p_offset_y+p_offset_y_added)+p_center_x*p_coeff_yx+p_center_x*p_coeff_yx_quad+p_coeff_y*p_center_y+p_center_y*p_coeff_y_quad+p_coeff_yyx*p_center_y*p_center_x;

    p_center_y=p_center_x_mod;
    p_center_x=-p_center_y_mod;


    std::cout << "2. p_center_x: " << p_center_x << "/ p_center_y: " << p_center_y << std::endl;
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
