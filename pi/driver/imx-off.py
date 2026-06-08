import time
import gpiod
from gpiod.line import Direction, Value
import sys

PINS = {
    "SEQ":  26,
    "LS":   6,
    "CLK":  5,
    "XCLR": 25,
}

config = {
    pin: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE, active_low=False)
    for pin in PINS.values()
}

with gpiod.Chip("/dev/gpiochip0") as chip:
    with chip.request_lines(config, consumer="sequence") as req:

        # Power-off sequence

        req.set_value(PINS["XCLR"], Value.INACTIVE)
        time.sleep(0.1)

        req.set_value(PINS["CLK"], Value.INACTIVE)
        time.sleep(0.1)

        req.set_value(PINS["LS"], Value.INACTIVE)
        time.sleep(0.1)

        req.set_value(PINS["SEQ"], Value.INACTIVE)
        time.sleep(0.1)

        print("IMX385 Powered Off")
        sys.exit(0)
