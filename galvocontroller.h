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

namespace Ui {
class OCTGalvosForm;
}

class GalvoController : public QWidget
{
    Q_OBJECT

public:
    explicit GalvoController();
    ~GalvoController();
private slots:
    void updateInfo(void);
    void displayFileNumber(int block_number);
    void scanTypeChosen(const QString& text);
    void startScan(void);
    void stopScan(void);
    void moveUp(void);
    void moveDown(void);
    void moveRight(void);
    void moveLeft(void);
    void goHome(void);
    void setSaveDir(void);
    void addDefaultScan(void);
    void setDefaultScan(void);
    void clearCurrentScan(void);
private:
    Ui::OCTGalvosForm *ui;
    float p_center_x;
    float p_center_y;
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
};



#endif // GalvoController_H
