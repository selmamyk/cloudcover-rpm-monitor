# Cloud Capture Modes

This folder contains a simple mode-based setup for running
`auto_capture_cloud.py` on a Raspberry Pi.

## Modes

### manual

Boot automation is disabled.

Use this when you want to run `auto_capture_cloud.py` manually from the
terminal.

### boot

The Pi runs `auto_capture_cloud.py` automatically after boot and keeps taking
images according to the configured `--interval` and `--num-images` arguments.

Default behavior in `capture_mode_config.json`:

- powers the IMX385 sensor on
- runs continuous capture with `--num-images 0`
- saves to `../../dataset/boot`
- writes HYTA CSV rows
- uses `--hyta-method imx385_adapted`

### low_power

The Pi runs `auto_capture_cloud.py` once after boot, schedules an RTC wake, and
then halts. On the next RTC wake, systemd runs the service again.

Default behavior in `capture_mode_config.json`:

- powers the IMX385 sensor on
- captures one image with `--num-images 1`
- saves to `../../dataset/low_power`
- writes HYTA CSV rows
- uses `--hyta-method imx385_adapted`
- schedules the next wake after `3600` seconds
- halts the Pi

## Set Mode

Run these commands on the Raspberry Pi:

```bash
cd ~/Cloud-cover/pi/cloud-software/auto_boot
chmod +x configure_capture_mode.sh
```

Disable boot automation:

```bash
./configure_capture_mode.sh manual
```

Run capture automatically after boot:

```bash
./configure_capture_mode.sh boot
```

Run one capture per boot and sleep between images:

```bash
./configure_capture_mode.sh low_power
```

Check the current setup:

```bash
./configure_capture_mode.sh status
```

## Test Without Reboot

After selecting `boot` or `low_power`, test the systemd service manually:

```bash
sudo systemctl start cloud-capture-mode.service
```

For `low_power`, test without shutdown first:

```bash
python3 capture_mode_runner.py --mode low_power --no-shutdown
```

## Logs

Systemd logs:

```bash
journalctl -u cloud-capture-mode.service -n 100 --no-pager
```

Mode runner log:

```bash
cat ../../boot_log/cloud_capture_mode.log
```

## Change Capture Settings

Edit:

```text
capture_mode_config.json
```

For normal boot capture, change:

```json
"boot": {
  "auto_capture_args": [...]
}
```

For low-power capture, change:

```json
"low_power": {
  "wake_seconds": 3600,
  "auto_capture_args": [...]
}
```

