#include <iostream>
#include <string>
#include "config.h"
#include "thorlabsrotation.h"
#include "Thorlabs.MotionControl.TCube.DCServo.h"
//#include "C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.TCube.DCServo.h"

// Exemple Thorlabs
//#include "stdafx.h"
//#include <stdlib.h>
//#include <conio.h>
//
//#if defined TestCode
//#include "..\..\..\Instruments\ThorLabs.TCube.DCServo\ThorLabs.TCube.DCServo\Thorlabs.MotionControl.TCube.DCServo.h"
//#else
//    #include "Thorlabs.MotionControl.TCube.DCServo.h"
//#endif

// TODO: constructor
ThorlabsRotation::ThorlabsRotation()
{
    // Initialization
    is_open = false;
    sprintf_s(serialNo, "%d", ROTATION_SERIAL_NUMBER);
    std::cout << "Initializing the Thorlabs Rotation Stage, serialNo=" << serialNo << std::endl;
}

// TODO: destructor
ThorlabsRotation::~ThorlabsRotation()
{
    if (is_open) {
        disconnect();
    }
}

// TODO: test information
void ThorlabsRotation::TestGetInformation()
{
    // Example_TDC001.cpp : Defines the entry point for the console application.
    int serialNo = 83828151; // TDC001

    // identify and access device
    char testSerialNo[16];
    sprintf_s(testSerialNo, "%d", serialNo);

    // Build list of connected device
    if (TLI_BuildDeviceList() == 0)
    {
        // Get device list size
        short n = TLI_GetDeviceListSize();
        std::cout<<"(DEBUG) DeviceListSize: "<< n <<std::endl;
        
        // Get TDC serial numbers
        char serialNos[100];
        TLI_GetDeviceListByTypeExt(serialNos, 100, 83);
        std::cout<<"(DEBUG) DeviceListByTypeExt: " << serialNos << std::endl;

        // Output list of matching devices
        {
            char *searchContext = nullptr;
            char *p = strtok_s(serialNos, ",", &searchContext);

            while (p != nullptr)
            {
                TLI_DeviceInfo deviceInfo;
                // Get device info from device
                TLI_GetDeviceInfo(p, &deviceInfo);
                // Get strings from device info structure
                char desc[65];
                strncpy_s(desc, deviceInfo.description, 64);
                desc[64] = '\0';
                char serialNo[9];
                strncpy_s(serialNo, deviceInfo.serialNo, 8);
                serialNo[8] = '\0';
                // output
                std::cout << "Found Device "<<p<<"="<<serialNo<<", "<<desc<<std::endl;
                p = strtok_s(nullptr, ",", &searchContext);
            }
        }
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
    // Stop polling
    CC_StopPolling(serialNo);

    // Disconnect and close the device.
    CC_Close(serialNo);

    // Disconnect from the simulation
    if (ROTATION_SIMULATION){
        TLI_UninitializeSimulations();
    }

    // Set state as close
    is_open = false;
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

// TODO: move_absolute
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
        std::cout << "Identifying the Thorlabs Rotation Stage" << std::endl;
        CC_ClearMessageQueue(serialNo);
        CC_Identify(serialNo);
    } else {
        std::cout << "Device is not open, can't identify." << std::endl;
    }
}

// TODO: initialize jog paramters reading the current values

void ThorlabsRotation::example_thorlabs(){
    // Example_TDC001.cpp : Defines the entry point for the console application.
    //
    std::cout<<"Running the Thorlabs DCMotor Example"<<std::endl;
    //int serialNo = 83837825;
    int serialNo = ROTATION_SERIAL_NUMBER;

    // set parameters
    int position = 15;
    int velocity = 0;

    // identify and access device
    char testSerialNo[16];
    sprintf_s(testSerialNo, "%d", serialNo);

    // Build list of connected device
    if (TLI_BuildDeviceList() == 0)
    {
        // get device list size
        short n = TLI_GetDeviceListSize();
        // get TDC serial numbers
        char serialNos[100];
        TLI_GetDeviceListByTypeExt(serialNos, 100, 83);

        // output list of matching devices
        {
            char *searchContext = nullptr;
            char *p = strtok_s(serialNos, ",", &searchContext);

            while (p != nullptr)
            {
                TLI_DeviceInfo deviceInfo;
                // get device info from device
                TLI_GetDeviceInfo(p, &deviceInfo);
                // get strings from device info structure
                char desc[65];
                strncpy_s(desc, deviceInfo.description, 64);
                desc[64] = '\0';
                char serialNo[9];
                strncpy_s(serialNo, deviceInfo.serialNo, 8);
                serialNo[8] = '\0';
                // output
                printf("Found Device %s=%s : %s\r\n", p, serialNo, desc);
                p = strtok_s(nullptr, ",", &searchContext);
            }
        }

        std::cout << testSerialNo << std::endl;

        // open device
        if(CC_Open(testSerialNo) == 0)
        {
            // start the device polling at 200ms intervals
            CC_StartPolling(testSerialNo, 200);

            Sleep(3000);
            // Home device
            CC_ClearMessageQueue(testSerialNo);
            CC_Home(testSerialNo);
            printf("Device %s homing\r\n", testSerialNo);

            // wait for completion
            WORD messageType;
            WORD messageId;
            DWORD messageData;
            CC_WaitForMessage(testSerialNo, &messageType, &messageId, &messageData);
            while(messageType != 2 || messageId != 0)
            {
                CC_WaitForMessage(testSerialNo, &messageType, &messageId, &messageData);
            }

            // set velocity if desired
            if(velocity > 0)
            {
                int currentVelocity, currentAcceleration;
                CC_GetVelParams(testSerialNo, &currentAcceleration, &currentVelocity);
                CC_SetVelParams(testSerialNo, currentAcceleration, velocity);
            }

            // move to position (channel 1)
            CC_ClearMessageQueue(testSerialNo);
            CC_MoveToPosition(testSerialNo, position);
            printf("Device %s moving\r\n", testSerialNo);

            // wait for completion
            CC_WaitForMessage(testSerialNo, &messageType, &messageId, &messageData);
            while(messageType != 2 || messageId != 1)
            {
                CC_WaitForMessage(testSerialNo, &messageType, &messageId, &messageData);
            }

            // get actual poaition
            int pos = CC_GetPosition(testSerialNo);
            printf("Device %s moved to %d\r\n", testSerialNo, pos);

            // stop polling
            CC_StopPolling(testSerialNo);
            // close device
            CC_Close(testSerialNo);
        }
    }
}

