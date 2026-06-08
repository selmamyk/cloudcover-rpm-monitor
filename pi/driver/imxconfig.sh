#!/bin/bash
set -e

run_step() {
    echo "[INFO] $*"
    timeout 10s "$@"
}

echo "[INFO] Loading imx385 driver..."
sudo modprobe imx385

echo "[INFO] Scanning media devices for imx385..."
for i in 0 1 2; do
    if media-ctl -d /dev/media$i -p | grep -q "imx385"; then
        MEDIA="media$i"
        echo "[INFO] Found imx385 on /dev/$MEDIA"
        break
    fi
done

if [ -z "$MEDIA" ]; then
    echo "[ERROR] imx385 not found on any /dev/media*"
    exit 1
fi

export MEDIA
echo "[INFO] Setting up media links for controller: ${MEDIA}"

run_step media-ctl -d /dev/${MEDIA} -l '"imx385 11-001a":0->"csi2":0[1]'
run_step media-ctl -d /dev/${MEDIA} -l '"csi2":4->"rp1-cfe-csi2_ch0":0[1]'
run_step media-ctl -d /dev/${MEDIA} -V '"imx385 11-001a":0 [fmt:SRGGB10_1X10/1920x1080 field:none]'
run_step media-ctl -d /dev/${MEDIA} -V '"csi2":0 [fmt:SRGGB10_1X10/1920x1080 field:none]'
run_step media-ctl -d /dev/${MEDIA} -V '"csi2":4 [fmt:SRGGB10_1X10/1920x1080 field:none]'

echo "[INFO] Media pipeline setup complete."
