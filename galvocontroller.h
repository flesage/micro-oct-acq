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
    float readCoeffX(void);
    float readCoeffY(void);
    void updateOffsetViewerX(void);
    void updateOffsetViewerY(void);
    void updateCoeffViewerX(void);
    void updateCoeffViewerY(void);
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
private:
    Ui::OCTGalvosForm *ui;
    QString dataDir;
    float p_center_x;
    float p_center_y;
    float p_offset_x;
    float p_offset_y;
    float p_coeff_x;
    float p_coeff_y;
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
};



#endif // GalvoController_H
