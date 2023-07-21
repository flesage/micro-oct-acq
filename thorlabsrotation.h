#ifndef THORLABSROTATION_H
#define THORLABSROTATION_H

#include <QWidget>
#include <QMutex>


class ThorlabsRotation : public QWidget
{
    Q_OBJECT

public:
    ThorlabsRotation();
    virtual ~ThorlabsRotation();
    void connect();
    void disconnect();
    void move_home();
    void move_jog_forwards();
    void move_jog_backwards();
    void move_absolute(float pos);
    void wait_for_completion();
    void stop();
    void stop_immediately();
    void set_jog_parameters(float step_size, float acceleration, float max_velocity);
    float get_position();
    void identify();
    void configure_ocrt(int n_angles);
signals:
    void sig_start_scan();
    void sig_change_filename(QString);

private slots:
    void slot_ocrt_next_position();

private:
    bool is_open;
    char serialNo[9];
    int p_n_ocrt_angles;
    int p_current_ocrt_angle;
    bool p_next_ocrt_asked;
    QMutex p_mutex;

};

#endif // THORLABSROTATION_H
