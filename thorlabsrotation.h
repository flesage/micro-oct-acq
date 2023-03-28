#ifndef THORLABSROTATION_H
#define THORLABSROTATION_H

class ThorlabsRotation
{
public:
    ThorlabsRotation();
    virtual ~ThorlabsRotation();
    void connect();
    void disconnect();
    void move_home();
    void move_jog_forwards();
    void move_jog_backwards();
    void move_absolute(float pos);
    void stop();
    void stop_immediately();
    void set_jog_parameters(float step_size, float acceleration, float max_velocity);
    float get_position();
    void identify();
private:
    bool is_open;
    char serialNo[9];
};

#endif // THORLABSROTATION_H
