#ifndef CONFIG_H
#define CONFIG_H

#define GALVOS_DEV "/OCT"
#define GALVOS_AOX "ao0"
#define GALVOS_AOY "ao1"

#define CAMERA_CLOCK "ctr0"
#define CAMERA_CLOCK_PFI "PFI12"

#define AICHANNELS "/OCT/ai0:1"
#define AIAOSAMPRATE 10000
#define N_AI_CHANNELS 2

#define NUM_GRAB_BUFFERS 3

#define LINE_ARRAY_SIZE 2048

// Rotation Stage Configuration (Thorlabs TDC001 + PRM1/MZ8)
// Communication Protocol: https://www.thorlabs.com/Software/Motion%20Control/APT_Communications_Protocol.pdf
#define ROTATION_SIMULATION false
#define ROTATION_SERIAL_NUMBER 83828151
#define ROTATION_STAGE_PORT_COM "COM3" // Joel's laptop: COM3
#define ROTATION_DEG2ENC 1919.6418578623391 // for the PRM1-Z8, Thorlabs communication protocol, section 8
#define ROTATION_VEL2ENC 42941.66 // deg/s, Thorlabs communication protocol, section 8
#define ROTATION_ACC2ENC 14.66 // deg/s2, Thorlabs communication protocol, section 8
#define ROTATION_REFRESH_RATE 200 // ms

#define SIMULATION

// OCT Server Configuration
#define SERVER_SOCKET_PORT 65432

#endif // CONFIG_H
