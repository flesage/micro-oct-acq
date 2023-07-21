#include <iostream>
#include <string>
#include "config.h"
#include "thorlabsrotation.h"
#include "Thorlabs.MotionControl.TCube.DCServo.h"

// Constructor
ThorlabsRotation::ThorlabsRotation() :
     p_n_ocrt_angles(4), p_current_ocrt_angle(0), p_next_ocrt_asked(false)
{
    // Initialization
    is_open = false;
    sprintf_s(serialNo, "%d", ROTATION_SERIAL_NUMBER);
}

// Destructor
ThorlabsRotation::~ThorlabsRotation()
{
    if (is_open) {
        disconnect();
    }

    // Disconnect from the simulation
    if (ROTATION_SIMULATION){
        TLI_UninitializeSimulations();
    }
}

// Open and connect to the kinesis device for communication
void ThorlabsRotation::connect()
{
    // Connect to the simulation
    if (ROTATION_SIMULATION){
        TLI_InitializeSimulations();
    } else {
        TLI_BuildDeviceList();
    }

    // Open the device for communication
    int error_code = CC_Open(serialNo);
    if (error_code == 0) {
        // Start polling position and stage
        CC_StartPolling(serialNo, ROTATION_REFRESH_RATE);

        // Set state as open
        is_open = true;
    } else {
        std::cout << "Unable to connect to the device. Error code: " << error_code << std::endl;
        is_open = false;
    }
}

// Disconnect and close the device
void ThorlabsRotation::disconnect()
{
    if (is_open == true) {
        // Stop polling
        CC_StopPolling(serialNo);

        // Disconnect and close the device.
        CC_Close(serialNo);

        // Set state as close
        is_open = false;
    }
}
    

// Move the kinesis device to its home
void ThorlabsRotation::move_home()
{
    if (is_open){
        CC_Home(serialNo);
    }
}

// Jog forwards
void ThorlabsRotation::move_jog_forwards()
{
    if (is_open) {
        CC_MoveJog(serialNo, MOT_Backwards); // For the rotation stage used, +deg is backward
    }
}

// Jog backwards
void ThorlabsRotation::move_jog_backwards()
{
    if (is_open) {
        CC_MoveJog(serialNo, MOT_Forwards); // For the rotation stage used, +deg is backward
    }
}

// Move (absolute)
void ThorlabsRotation::move_absolute(float pos_deg)
{
    if (is_open){
        int pos_enc = pos_deg * ROTATION_DEG2ENC;
        CC_SetMoveAbsolutePosition(serialNo, pos_enc);
        CC_MoveAbsolute(serialNo);
    }
}

void ThorlabsRotation::stop()
{
    if (is_open){
        CC_StopProfiled(serialNo);
    }
}

void ThorlabsRotation::stop_immediately()
{
    if (is_open) {
        CC_StopImmediate(serialNo);
    }
}

void ThorlabsRotation::set_jog_parameters(float step_size, float acceleration, float max_velocity)
{
    if (is_open){
        CC_SetJogStepSize(serialNo, int(step_size*ROTATION_DEG2ENC));
        CC_SetJogVelParams(serialNo, int(acceleration*ROTATION_ACC2ENC), int(max_velocity * ROTATION_VEL2ENC));
    }
}

float ThorlabsRotation::get_position()
{
    if (is_open) {
        int pos_enc = CC_GetPosition(serialNo);
        float pos_deg = float(pos_enc) / ROTATION_DEG2ENC;
        return pos_deg;
    }
    return 0.0f;
}

void ThorlabsRotation::identify()
{
    if (is_open){
        CC_Identify(serialNo);
    } else {
        std::cout << "Device is not open, can't identify." << std::endl;
    }
}

void ThorlabsRotation::wait_for_completion()
{
    if (is_open) {
        std::cerr << "Waiting for completion... ";
        WORD messageType;
        WORD messageId;
        DWORD messageData;

        CC_WaitForMessage(serialNo, &messageType, &messageId, &messageData);
        while(messageType != 2 || messageId != 1)
        {
            CC_WaitForMessage(serialNo, &messageType, &messageId, &messageData);
        }
        std::cerr << "done!" << std::endl;
    }
}

void ThorlabsRotation::configure_ocrt(int n_angles){
    p_n_ocrt_angles = n_angles;
    p_current_ocrt_angle = 0;
}

void ThorlabsRotation::slot_ocrt_next_position(){
    p_mutex.lock();
    if (p_next_ocrt_asked) {
        p_mutex.unlock();
        return;
    } else {
        p_next_ocrt_asked = true;
        p_mutex.unlock();
    }

    if (p_current_ocrt_angle == p_n_ocrt_angles){
        // emit ocrt_acquisition_done;
        return;
    }

    float angle = 360 / p_n_ocrt_angles * p_current_ocrt_angle;
    std::cerr << "Moving to angle=" << angle << std::endl;
    move_absolute(angle);
    wait_for_completion(); // FIXME: this freezes the GUI.

    // Change the filename
    char buffer [20];
    snprintf(buffer, 20, "ocrt_deg%02.4f", angle);
    QString fileName = QString(buffer);
    emit sig_change_filename(fileName);

    // Update the next slice
    p_current_ocrt_angle++;

    // Request an acquisition
    emit sig_start_scan();
    p_mutex.lock();
    p_next_ocrt_asked = false;
    p_mutex.unlock();
}
