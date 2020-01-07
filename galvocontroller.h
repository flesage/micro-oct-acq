#ifndef GalvoController_H
#define GalvoController_H

#include <QWidget>
#include <QDir>
#include <QSettings>
#include <QStringList>
#include <QTimer>

#include "galvos.h"
#include "camera.h"
#include "datasaver.h"
#include "fringeviewer.h"
#include "imageviewer.h"
#include "datasaver.h"
#include "float64datasaver.h"
#include "analoginput.h"
#include "motorclass.h"

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
private slots:
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
    void moveUp(void);
    void moveDown(void);
    void moveRight(void);
    void moveLeft(void);
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
    void addDefaultScan(void);
    void setDefaultScan(void);
    void clearCurrentScan(void);
    void slot_updateImageThreshold();
    void slot_updateHanningThreshold();
    void slot_openMotorPort(bool flag);
    void slot_doMosaic();
    void slot_doStack();
    void updateCenterLineEdit(void);
    void automaticCentering(void);
    QString readLineNumber(void);
private:
    Ui::OCTGalvosForm *ui;
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
    QString p_line_number_str;
    Galvos p_galvos;
    Camera* p_camera;
    QDir p_save_dir;
    QSettings p_settings;
    QStringList p_saved_scans;
    FringeViewer* p_fringe_view;
    ImageViewer* p_image_view;
    DataSaver* p_data_saver;
    QTimer* view_timer;
    AnalogInput* p_ai;
    Float64DataSaver* p_ai_data_saver;
    int p_block_size;
    MotorClass* motors;
    bool flagMotor;

};



#endif // GalvoController_H
