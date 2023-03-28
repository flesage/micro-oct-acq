#include <iostream>
#include <string>
#include "config.h"
#include "thorlabsrotation.h"
#include "Thorlabs.MotionControl.TCube.DCServo.h"

// Constructor
ThorlabsRotation::ThorlabsRotation()
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