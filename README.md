# Cloud Cover RPM Monitor

This repository contains the Raspberry Pi software and evaluation code for a
combined cloud-cover and wind-turbine RPM monitoring system.

The system can:

- capture images with an IMX385 camera
- estimate cloud cover using HYTA-based image processing
- measure wind-turbine RPM from video or live camera input
- show cloud-cover or RPM status on a Waveshare 2.66" e-paper display
- run cloud capture automatically after boot, including a low-power mode
- evaluate cloud-cover methods on local datasets

For detailed hardware setup, installation steps and operating instructions see the project "Complete Setup and Operation Manual.

## Repository Layout

```text
.
|-- pi/
|   |-- cloud-software/          # Cloud-cover capture and HYTA prediction
|   |-- wind-turbine-software/   # RPM measurement, video recording, config tools
|   |-- driver/                  # IMX385 driver, overlay, GPIO, and media setup
|   `-- e-Paper/                 # Waveshare e-paper Python library
|-- cloud-cover-evaluation/      # Cloud-cover method evaluation and training code
|-- LICENSE
`-- README.md
```

## Raspberry Pi Location

On the Raspberry Pi, the repository is expected to be located at:

```bash
~/Cloud-cover
```

This path is important because the display code uses the bundled Waveshare
library from:

```text
/home/cloudpi/Cloud-cover/pi/e-Paper/RaspberryPi_JetsonNano/python/lib
```

## Main Components

### Cloud-Cover Capture

Main script:

```text
pi/cloud-software/auto_capture_cloud.py
```

This script captures images, processes them with HYTA-based methods, stores
results, and can optionally update the e-paper display.

Autostart and low-power capture modes are configured in:

```text
pi/cloud-software/auto_boot/
```

Additional notes for capture modes are available in:

```text
pi/cloud-software/auto_boot/CAPTURE_MODES.md
```

### Wind-Turbine RPM

Main script:

```text
pi/wind-turbine-software/run_wind_turbine.py
```

This script measures RPM from either live camera input or a video file. It
supports BPM and optical-flow based processing, CSV logging, preview output,
processed video output, and optional e-paper display updates.

Related tools:

```text
pi/wind-turbine-software/record_video.py
pi/wind-turbine-software/config_generator.py
```

### Camera Driver and Pi Hardware

Driver and hardware setup files are stored in:

```text
pi/driver/
```

This folder contains the IMX385 driver files, overlay files, GPIO helper scripts,
and media pipeline setup script.

### E-Paper Display

The bundled Waveshare e-paper library is stored in:

```text
pi/e-Paper/RaspberryPi_JetsonNano/python/lib
```

Both the cloud-cover and RPM programs use this library when e-paper display
updates are enabled.

### Cloud-Cover Evaluation

Evaluation and training code is stored in:

```text
cloud-cover-evaluation/
```

This includes shared cloud-cover method code, evaluation scripts, parameter
search scripts, and Tiny U-CloudNet training/inference code. 

## Typical Workflow

1. Place the repository on the Raspberry Pi as `~/Cloud-cover`.
2. Follow the User Manual for hardware wiring, dependencies, camera setup, and
   display setup.
3. Use `pi/cloud-software/` for cloud-cover capture.
4. Use `pi/wind-turbine-software/` for RPM measurement and video recording.
5. Use `pi/cloud-software/auto_boot/` to configure boot or low-power operation.
6. Use `cloud-cover-evaluation/` on a development machine for analysis,
   experiments, and model evaluation.


## License

See `LICENSE`.
