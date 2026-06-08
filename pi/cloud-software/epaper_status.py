from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


DEFAULT_EPAPER_LIB_PATH = "/home/cloudpi/Cloud-cover/pi/e-Paper/RaspberryPi_JetsonNano/python/lib"


def update_cloud_status_display(
    cloud_percent: float,
    okta: int,
    prediction_time: datetime,
    epaper_lib_path: str = DEFAULT_EPAPER_LIB_PATH,
) -> None:
    if epaper_lib_path and epaper_lib_path not in sys.path:
        sys.path.append(epaper_lib_path)

    from PIL import Image, ImageDraw, ImageFont
    from waveshare_epd import epd2in66

    epd = epd2in66.EPD()
    epd.init(0)
    epd.Clear()

    image = Image.new("1", (epd.height, epd.width), 255)
    draw = ImageDraw.Draw(image)

    font_large = _load_font(28, bold=True)
    font_medium = _load_font(22, bold=True)
    font_small = _load_font(16, bold=False)

    draw.rectangle((0, 0, epd.height - 1, epd.width - 1), outline=0, width=2)
    draw.text((12, 10), "Cloud mode", font=font_large, fill=0)
    draw.line((10, 46, epd.height - 10, 46), fill=0, width=2)

    draw.text((12, 58), "Cloud cover", font=font_small, fill=0)
    draw.text((12, 78), f"{cloud_percent:.1f}%", font=font_medium, fill=0)
    draw.text((142, 82), f"Okta {okta}/8", font=font_small, fill=0)

    draw.text((12, 116), "Last prediction", font=font_small, fill=0)
    draw.text((12, 136), prediction_time.strftime("%d.%m.%Y %H:%M"), font=font_small, fill=0)

    epd.display(epd.getbuffer(image))
    epd.sleep()


def _load_font(size: int, bold: bool) -> object:
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else "",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()
