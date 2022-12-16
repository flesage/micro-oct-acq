#ifndef THORLABSROTATION_H
#define THORLABSROTATION_H

class ThorlabsRotation
{
public:
    ThorlabsRotation();
    virtual ~ThorlabsRotation();
    void TestGetInformation();
    void connect();
    void disconnect();
    void move_home();
    void move_jog();
    void move_absolute(float pos);
    void set_jog_parameters()
};

#endif // THORLABSROTATION_H
