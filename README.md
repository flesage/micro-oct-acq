# micro-oct-acq
Code used to control acquisition for microscopic oct

## Dependencies
* NI-Max
  * NI-IQAMdx
* Qt6
  * QtSerialPort has to be explicitly installed.
* ArrayFire : https://arrayfire.com/

## Development

* Setup a virtual acquisition card with `NI MAX`
  * Open the `NI MAX` software
  * Right click on `Périphériques et interfaces` and choose `Créer un nouvel objet...`
  * Choose the option: `Périphérique ou instrument modulaire NI-DAQmx simulé`
  * Wait for the simulated device list to appear, and choose one of the `usb-x e.g. 6353`
  * Use the same device name as in the `config.h` file (e.g. `#define GALVOS_DEV "/OCT`)
* COnfigure arrayfire