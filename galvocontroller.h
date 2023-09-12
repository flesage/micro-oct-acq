#ifndef GalvoController_H
#define GalvoController_H

#include <QWidget>
#include <QDir>
#include <QSettings>
#include <QStringList>
#include <QTimer>
#include <QMutex>
#include "galvos.h"
#include "camera.h"
#include "softwarecamera.h"
#include "datasaver.h"
#include "saver_image.h"
#include "saver_remote.h"
#include "fringeviewer.h"
#include "imageviewer.h"
#include "float64datasaver.h"
#include "analoginput.h"
#include "motorclass.h"
#include "thorlabsrotation.h"
#include "oct3dorthogonalviewer.h"
#include "octserver.h"

namespace Ui {
class OCTGalvosForm;
}

class GalvoController : public QWidget
{
    Q_OBJECT

public:
    explicit GalvoController();
    ~GalvoController();
signals:
    void sig_updateImageThreshold(float);
    void sig_updateHanningThreshold(float);
    void sig_updateViewLinePositions(bool,int,int);
    void sig_updateAveragingFlag(bool);
    void sig_updateAveragingAlgo(int);
    void sig_serverEndScan();
    void sig_serverEndScanAndSendImage(int, int, int, float*);
    void sig_serverStartTransfer();

private slots:
    void updateExposure(int exposure_us);
    void updateSpeedPiezo(void);
    void turnPiezoOn(void);
    void turnPiezoOff(void);
    void homePiezo(void);
    void jogPiezo(void);
    void stopPiezo(void);
    void invertAxes(void);
    void updateInfo(void);
    void checkPath(void);
    void autoFillName(void);
    void displayFileNumber(int block_number);
    void scanTypeChosen(const QString& text);
    void startScan(void);
    void stopScan(void);
    void stopFiniteScan(void);
    void moveUp(void);
    void moveDown(void);
    void moveRight(void);
    void moveLeft(void);
    void moveMotorUp(void);
    void moveMotorDown(void);
    void moveMotorYP(void);
    void moveMotorYM(void);
    void moveMotorXP(void);
    void moveMotorXM(void);
    void goMotorHome(void);
    void readCoeffTxt(void);
    void writeCoeffTxt(void);
    void readOffset(void);
    float readOffsetX(void);
    float readOffsetY(void);
    float readOffsetX_2(void);
    float readOffsetY_2(void);
    void updateOffsetViewerX(void);
    void updateOffsetViewerY(void);
    void updateOffset(void);
    void goHome(void);
    void setSaveDir(void);
    void setFileName(QString);
    void addDefaultScan(void);
    void setDefaultScan(void);
    void clearCurrentScan(void);
    void slot_updateImageThreshold();
    void slot_updateHanningThreshold();
    void slot_openMotorPort(bool flag);
    void slot_setStagePosition(float x, float y, float z);
    void slot_doMosaic();
    void slot_doStack();
    void updateCenterLineEdit(void);
    void automaticCentering(void);
    QString readLineNumber(void);
    void setLineScanPos(int start_x, int start_y, int stop_x, int stop_y);
    void setFixedLengthLineScanPos(void);
    void slot_updateViewLinePositions(void);
    void slot_updateAverageAngiogram(void);
    void slot_updateAngiogramAlgo(void);
    void setCenterFromLineScan(void);

   // Rotation stage
    void activateRotationStage(bool flag);
    void identifyRotationStage(void);
    void homeRotationStage(void);
    void slot_rotation_updateJogParameters(void);
    void slot_rotation_jogForward(void);
    void slot_rotation_jogReverse(void);
    void slot_rotation_absoluteMove(void);
    void slot_rotation_stop(void);
    void slot_rotation_stop_immediately(void);
    void slot_rotation_update_position(void);

    // Server
    void slot_server(void);
    void slot_server_set_type(QString);

private:
    Ui::OCTGalvosForm *ui;
    //OCTServer *server;
    QString dataDir;
    float p_center_x;
    float p_center_y;
    float p_offset_x;
    float p_offset_y;
    float p_coeff_x;
    float p_coeff_y;
    float p_coeff_x_quad;
    float p_coeff_y_quad;
    float p_coeff_xy;
    float p_coeff_yx;
    float p_coeff_xy_quad;
    float p_coeff_yx_quad;
    float p_coeff_yyx;
    float p_coeff_xxy;
    float p_offset_x_added;
    float p_offset_y_added;
    float p_offset_x_added_2;
    float p_offset_y_added_2;
    int p_line_number;
    float p_start_line_x;
    float p_stop_line_x;
    float p_start_line_y;
    float p_stop_line_y;
    float p_line_length;
    int p_start_viewline;
    int p_stop_viewline;
    QString p_line_number_str;
    Galvos p_galvos;
    bool p_finite_acquisition;
    bool p_stack_acquisition;
    int p_acq_index;
    int p_n_volumes;
    QString p_datasetname;
    OCTServer* p_server;
    bool p_server_mode;
    QString p_server_type;
    bool p_server_stop_asked;
#ifndef SIMULATION
    Camera* p_camera;
#else
    SoftwareCamera* p_camera;
#endif
    bool p_camera_stop_requested;
    QDir p_save_dir;
    QSettings p_settings;
    QStringList p_saved_scans;
    FringeViewer* p_fringe_view;
    ImageViewer* p_image_view;
    oct3dOrthogonalViewer* p_ortho_view;
    DataSaver* p_data_saver;
    SaverImage* p_image_saver;
    Saver_Remote* p_remote_saver;
    bool p_active_viewer;
    QTimer* view_timer;
    QTimer* rotation_timer;
    AnalogInput* p_ai;
    Float64DataSaver* p_ai_data_saver;
    int p_block_size;
    MotorClass* motors;
    ThorlabsRotation* thorlabs_rotation;
    bool flagMotor;
    float* p_disp_comp_vec_10x;
    float* p_disp_comp_vec_25x;
    QMutex p_mutex;
    bool p_serverTransferDone;
};



#endif // GalvoController_H
