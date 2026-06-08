import time
import gpiod
from gpiod.line import Direction, Value
import sys

PINS = {
    "HB+":  17,
    "HB-":  27,
}

config = {
    pin: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE, active_low=False)
    for pin in PINS.values()
}

with gpiod.Chip("/dev/gpiochip0") as chip:
    with chip.request_lines(config, consumer="shutter") as req:

        req.set_value(PINS["HB+"], Value.INACTIVE)
        req.set_value(PINS["HB-"], Value.ACTIVE)
        time.sleep(0.15)

        req.set_value(PINS["HB+"], Value.INACTIVE)
        req.set_value(PINS["HB-"], Value.INACTIVE)
        time.sleep(0.1)

        print("IR-Cut Filter Off")
        sys.exit(0)