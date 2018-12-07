#include "galvocontroller.h"
#include <iostream>
#include <QIntValidator>
#include <QDoubleValidator>
#include <QDir>
#include <QFileDialog>
#include <QTimer>
#include "ui_oct_galvos_form.h"
#include "config.h"

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
    QIntValidator* validator3=new QIntValidator(1,32,this);
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

    connect(ui->lineEdit_hanningeps,SIGNAL(editingFinished()),this,SLOT(slot_updateHanningThreshold()));
    connect(ui->lineEdit_logeps,SIGNAL(editingFinished()),this,SLOT(slot_updateImageThreshold()));
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
    ui->pushButton_start->setEnabled(true);
    ui->pushButton_stop->setEnabled(false);
}

GalvoController::~GalvoController()
{
    delete ui;
}

void GalvoController::setSaveDir()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("choose Directory"),
                                                    p_save_dir.absolutePath(),
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);
    p_save_dir.setPath(dir);
    ui->label_directory->setText(p_save_dir.absolutePath());
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
    QString tmp;
    tmp.sprintf("Current time per pixel: %10.2f us\n",time_per_pix);
    text = text+tmp;
    tmp.sprintf("Current exposure: %10.2f us\n",exposure);
    text=text+tmp;
    tmp.sprintf("Beam speed in x: %10.2f mm/sec\n",speed_x);
    text=text+tmp;

    if (time_per_pix < exposure)
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

void GalvoController::startScan()
{
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
    double radians_per_volt = 2*3.14159265359/(360*0.8);
    double f1=50.0;
    double f2=100.0;
    if(telescope == 1 || telescope == 2)
    {
        f1=75.0;
        f2=75.0;
    }

    int objective = ui->comboBox_objective->currentIndex();
    double fobj=18.0;
    switch(objective)
    {
    case 0:
        break;
    case 1:
        fobj=36.0;
        break;
    case 2:
        fobj=20.0;
        break;
    case 3:
        fobj=9.0;
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
        float dop_kernel = ui->lineEdit_doppler_spatial_kernel->text().toFloat();
        float spatial_kernel_size=ui->lineEdit_doppler_spatial_kernel->text().toFloat();
        float line_period = 1.0f/line_rate/(nx+n_extra);
        float dimx = width/nx;
        float dimz = 3.5;
        p_image_view = new ImageViewer(0,nx+n_extra,view_depth,hpf_time_constant,line_period,spatial_kernel_size,dimz,dimx);
        p_image_view->updateHanningThreshold(ui->lineEdit_hanningeps->text().toFloat());
        p_image_view->updateImageThreshold(ui->lineEdit_logeps->text().toFloat());
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
}
