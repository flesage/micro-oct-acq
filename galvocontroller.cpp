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

GalvoController::GalvoController() :
    ui(new Ui::OCTGalvosForm), p_galvos(GALVOS_DEV,GALVOS_AOX,GALVOS_AOY), p_settings("Polytechnique/LIOM","OCT"),
    p_ai(0), p_ai_data_saver(0)
{
    ui->setupUi(this);
    p_center_x=0.0;
    p_center_y=0.0;
    p_block_size = 256;
    p_camera = 0;
    p_fringe_view = 0;
    p_image_view = 0;
    p_data_saver = 0;

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
    QIntValidator* validator4=new QIntValidator(1,200,this);
    ui->lineEdit_aline_repeat->setValidator(validator4);
    QIntValidator* validator5=new QIntValidator(10,2048,this);
    ui->lineEdit_displayPoints->setValidator(validator5);

    connect(ui->lineEdit_linerate,SIGNAL(editingFinished()),this,SLOT(updateInfo()));
    connect(ui->lineEdit_nx,SIGNAL(editingFinished()),this,SLOT(updateInfo()));
    connect(ui->lineEdit_extrapoints,SIGNAL(editingFinished()),this,SLOT(updateInfo()));
    connect(ui->lineEdit_exposure,SIGNAL(editingFinished()),this,SLOT(updateInfo()));
    connect(ui->lineEdit_width,SIGNAL(editingFinished()),this,SLOT(updateInfo()));
    connect(ui->lineEdit_aline_repeat,SIGNAL(editingFinished()),this,SLOT(updateInfo()));

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
    connect(ui->horizontalScrollBar_coeffY,SIGNAL(valueChanged(int)),this,SLOT(updateCoeffViewerY()));
    connect(ui->horizontalScrollBar_coeffX,SIGNAL(valueChanged(int)),this,SLOT(updateCoeffViewerX()));

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
    ui->pushButton_start->setEnabled(true);
    ui->pushButton_stop->setEnabled(false);
    ui->lineEdit_datasetname->setEnabled(false);
    readCoeffTxt();
    updateOffsetViewerX();
    updateOffsetViewerY();
    updateCoeffViewerX();
    updateCoeffViewerY();
}

GalvoController::~GalvoController()
{
    delete motors;
    delete ui;
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
    for(unsigned int i=0; i<=nx; i++)
    {
        for(unsigned int j=0; j<=ny; j++)
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
    for(unsigned int i=0; i<=nz; i++)
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
    tmp.sprintf("Current time per pixel: \t %10.2f us\n",time_per_pix);
    text = text+tmp;
    tmp.sprintf("Current exposure: \t %10.2f us\n",exposure);
    text=text+tmp;
    tmp.sprintf("Beam speed in x: \t %10.2f mm/sec\n",speed_x);
    text=text+tmp;
    tmp.sprintf("Lateral sampling in x: \t %10.3f um/pix\n",lat_sampling);
    text=text+tmp;
    tmp.sprintf("Inter B-scan time in x: \t %10.3f ms\n",interFrameTime);
    text=text+tmp;

    if (time_per_pix < 0.9*exposure)
    {
        tmp.sprintf("WARNING: INTEGRATION TIME HIGHER THAN TIME PER PIXEL!!!\n");
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

    if (!overwriteFlag)
    {
        while(folderFlag)
        {
            if (pathToData.exists())
            {
                newfolderName = folderName+"_"+QString::number(counter);
                pathToData = QDir::cleanPath(dataDir + QDir::separator() + newfolderName +QDir::separator());
                counter++;
            }
            else
            {
                folderFlag=false;
                ui->lineEdit_datasetname->setText(newfolderName);
            }
        }
    }
}

//void GalvoController::startScanCheckPath()
//{
//    checkPath();
//    startScan();
//}


void GalvoController::startScan()
{
    checkPath();
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

    int telescope = ui->comboBox_telescope->currentIndex();
    double radians_per_volt = 2*3.14159265359/(360*0.8)*(3.0/4.0)/0.95;
    double f1=50.0;
    double f2=100.0;
    switch(telescope)
    {
    case 0:
        f1=54.0;
        f2=300.0;
        break;
    case 1:
        f1=50.0;
        f2=100.0;
        break;
    case 2:
        f1=75.0;
        f2=75.0;
        break;
    case 3:
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
    p_camera=new Camera(nx+n_extra,exposure);

    // If we are saving, setup for it
    if (ui->checkBox_save->isChecked())
    {
        p_block_size = (512*256)/nx;

        p_data_saver = new DataSaver(nx+n_extra,p_block_size);
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
        p_image_view = new ImageViewer(0,nx+n_extra,view_depth,n_repeat, hpf_time_constant,line_period,spatial_kernel_size,dimz,dimx);
        p_image_view->updateHanningThreshold(ui->lineEdit_hanningeps->text().toFloat());
        p_image_view->updateImageThreshold(ui->lineEdit_logeps->text().toFloat());
        p_image_view->updateAngioAlgo(ui->comboBox_angio->currentIndex());
        connect(view_timer,SIGNAL(timeout()),p_image_view,SLOT(updateView()));
        connect(this,SIGNAL(sig_updateHanningThreshold(float)),p_image_view,SLOT(updateHanningThreshold(float)));
        connect(this,SIGNAL(sig_updateImageThreshold(float)),p_image_view,SLOT(updateImageThreshold(float)));

        p_image_view->show();
        p_camera->setImageViewer(p_image_view);
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
        view_timer->start(100);
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
        p_galvos.setLineRamp(p_center_x-width/2,p_center_y-height/2,p_center_x+width/2,p_center_y+height/2,nx,ny,n_extra,line_rate);
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
}

void GalvoController::moveRight(void)
{
    p_center_x-=10;
    p_galvos.move(p_center_x,p_center_y);
}

void GalvoController::moveLeft(void)
{
    p_center_x+=10;
    p_galvos.move(p_center_x,p_center_y);
}

void GalvoController::goHome(void)
{
    p_center_x=0;
    p_center_y=0;
    p_galvos.move(p_center_x,p_center_y);
    std::cout << "p_center_x" << p_center_x << "/ p_center_y" << p_center_y << std::endl;
}

void GalvoController::writeCoeffTxt(void)
{
    std::cout<<"in readcoeffTxt"<<std::endl;
    QFile file("C:/git-projects/micro-oct-acq/coefficients.txt");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
    QTextStream out(&file);
    p_offset_x=readOffsetX();
    p_offset_y=readOffsetY();
    p_coeff_x=readCoeffX();
    p_coeff_y=readCoeffY();
    out << p_coeff_x << "/" << p_coeff_y << "/" << p_offset_x << "/" << p_offset_y << "\n";
    file.close();
}

void GalvoController::readCoeffTxt(void)
{
    std::cout<<"in readcoeffTxt"<<std::endl;
    QFile file("C:/git-projects/micro-oct-acq/coefficients.txt");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return;
    QTextStream in(&file);
    QString line = in.readLine();
    file.close();
    QStringList fields = line.split("/");
    p_coeff_x=fields.at(0).toFloat();
    p_coeff_x=p_coeff_x*10.0;
    ui->horizontalScrollBar_coeffX->setValue(p_coeff_x);

    p_coeff_y=fields.at(1).toFloat();
    p_coeff_y=p_coeff_y*10.0;
    ui->horizontalScrollBar_coeffY->setValue(p_coeff_y);

    p_offset_x=fields.at(2).toFloat();
    ui->horizontalScrollBar_offsetX->setValue(p_offset_x);

    p_offset_y=fields.at(3).toFloat();
    ui->horizontalScrollBar_offsetY->setValue(p_offset_y);

    std::cout<<"coeff:"<< p_coeff_x << "/" << p_coeff_y << "/" << p_offset_x << "/" << p_offset_y << "/" <<std::endl;
    //updateOffsetViewers();
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
    p_center_x=fields.at(1).toFloat();
    p_center_y=fields.at(0).toFloat();
    ui->lineEdit_readOffsetX->setText(QString::number(p_center_x));
    ui->lineEdit_readOffsetY->setText(QString::number(p_center_y));
    p_offset_x=readOffsetX();
    p_offset_y=readOffsetY();
    p_coeff_x=readCoeffX();
    p_coeff_y=readCoeffY();
    p_center_x=p_center_x*p_coeff_x+p_offset_x;
    p_center_y=p_center_y*p_coeff_y+p_offset_y;
    std::cout << "p_center_x" << p_center_x << "/ p_center_y" << p_center_y << std::endl;
    p_galvos.move(p_center_x,p_center_y);
}

void GalvoController::updateOffset(void)
{
    p_galvos.move(0,0);
    p_center_x=ui->lineEdit_readOffsetX->text().toFloat();
    p_center_y=ui->lineEdit_readOffsetY->text().toFloat();
    p_offset_x=readOffsetX();
    p_offset_y=readOffsetY();
    p_coeff_x=readCoeffX();
    p_coeff_y=readCoeffY();
    p_center_x=p_center_x*p_coeff_x+p_offset_x;
    p_center_y=p_center_y*p_coeff_y+p_offset_y;
    p_galvos.move(p_center_x,p_center_y);
}

void GalvoController::updateOffsetViewerX(void)
{
    p_offset_x=readOffsetX();
    ui->lineEdit_offsetX->setText(QString::number(p_offset_x));
    updateOffset();
    std::cout<<"in update: offsetX:"<< p_offset_x <<std::endl;
}

void GalvoController::updateOffsetViewerY(void)
{
    p_offset_y=readOffsetY();
    ui->lineEdit_offsetY->setText(QString::number(p_offset_y));
    updateOffset();
    std::cout<<"in update: offsetY:"<< p_offset_y <<std::endl;
}

void GalvoController::updateCoeffViewerX(void)
{
    p_coeff_x=readCoeffX();
    ui->lineEdit_coeffX->setText(QString::number(p_coeff_x));
    updateOffset();
    std::cout<<"in update: coeffX:"<< p_coeff_x <<std::endl;
}

void GalvoController::updateCoeffViewerY(void)
{
    p_coeff_y=readCoeffY();
    ui->lineEdit_coeffY->setText(QString::number(p_coeff_y));
    updateOffset();
    std::cout<<"in update: coeffY:"<< p_coeff_y << std::endl;
}

float GalvoController::readOffsetX(void)
{
    p_offset_x=ui->horizontalScrollBar_offsetX->value();
    return p_offset_x;
}

float GalvoController::readOffsetY(void)
{
    p_offset_y=ui->horizontalScrollBar_offsetY->value();
    return p_offset_y;
}

float GalvoController::readCoeffX(void)
{
    p_coeff_x=ui->horizontalScrollBar_coeffX->value();
    p_coeff_x=p_coeff_x/10;
    return p_coeff_x;
}

float GalvoController::readCoeffY(void)
{
    p_coeff_y=ui->horizontalScrollBar_coeffY->value();
    p_coeff_y=p_coeff_y/10;
    return p_coeff_y;
}
