from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


DEFAULT_EPAPER_LIB_PATH = "/home/cloudpi/e-Paper/RaspberryPi_JetsonNano/python/lib"


def update_rpm_status_display(
    rpm: float | str | None,
    update_time: datetime,
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
    font_medium = _load_font(24, bold=True)
    font_small = _load_font(16, bold=False)

    draw.rectangle((0, 0, epd.height - 1, epd.width - 1), outline=0, width=2)
    draw.text((12, 10), "RPM mode", font=font_large, fill=0)
    draw.line((10, 46, epd.height - 10, 46), fill=0, width=2)

    draw.text((12, 58), "Current RPM", font=font_small, fill=0)
    draw.text((12, 80), _format_rpm(rpm), font=font_medium, fill=0)

    draw.text((12, 118), "Last update", font=font_small, fill=0)
    draw.text((12, 138), update_time.strftime("%d.%m.%Y %H:%M"), font=font_small, fill=0)

    epd.display(epd.getbuffer(image))
    epd.sleep()


def _format_rpm(rpm: float | str | None) -> str:
    if rpm is None or rpm == "":
        return "No RPM"
    try:
        return f"{float(rpm):.2f}"
    except (TypeError, ValueError):
        return str(rpm)


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
