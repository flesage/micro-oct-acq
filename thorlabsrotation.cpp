#include <iostream>
#include <string>
#include "config.h"
#include "thorlabsrotation.h"
#include "Thorlabs.MotionControl.TCube.DCServo.h"

ThorlabsRotation::ThorlabsRotation()
{
    // Connect to the simulation
    if (ROTATION_SIMULATION){
        TLI_InitializeSimulations();
    }
}

ThorlabsRotation::~ThorlabsRotation()
{
    if (ROTATION_SIMULATION){
        TLI_UninitializeSimulations();
    }
}


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

void ThorlabsRotation::connect()
{
}

void ThorlabsRotation::disconnect()
{
}

void ThorlabsRotation::move_home()
{
}

void ThorlabsRotation::move_jog()
{
}

void ThorlabsRotation::move_absolute(float pos)
{
}
