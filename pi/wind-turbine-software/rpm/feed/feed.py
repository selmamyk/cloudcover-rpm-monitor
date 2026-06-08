import cv2 as cv
import numpy as np
from .sources import make_frame_source


class Feed:
    def __init__(self, **kwargs):
        self.crop_points = kwargs["crop_points"]
        self.adjust_contrast = False
        self.contrast_multiplier = kwargs.get("contrast_multiplier", 1.0)
        self.frame_cnt = 0
        self._set_base_config(kwargs, kwargs["fps"])

    def _set_base_config(self, params, fps) -> None:
        self.target = params.get("target")
        self.fps = fps
        self.video = make_frame_source(params)
        # self.video.set(cv.CAP_PROP_FPS, self.fps)

        if self.crop_points is not None:
            self.h = self.crop_points[0][1] - self.crop_points[0][0]
            self.w = self.crop_points[1][1] - self.crop_points[1][0]
            self.yrange = slice(self.crop_points[0][0], self.crop_points[0][1])
            self.xrange = slice(self.crop_points[1][0], self.crop_points[1][1])
        else:
            img = self.get_frame()
            self.h, self.w, self.ch = img.shape
            self.yrange = slice(0, self.h)
            self.xrange = slice(0, self.w)

    def get_frame(self) -> np.ndarray:
        ret, frame = self.video.read()
        self.isActive = ret
        if ret:
            self.frame_cnt += 1
        if self.crop_points is not None and ret:
            frame = frame[self.yrange, self.xrange]
        if self.adjust_contrast:
            frame = cv.convertScaleAbs(frame, alpha=self.contrast_multiplier)
        return frame


class RpmFromFeed(Feed):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_config_parameters(kwargs["crop_points"])

    def _set_config_parameters(self, crop_points) -> None:
        self.crop_points = crop_points

        self.radius_x = self.w // 2
        self.radius_y = self.h // 2
        if self.radius_x > self.radius_y:
            self.radius_max = self.radius_x
        else:
            self.radius_max = self.radius_y

        if (crop_points is not None) and (self.w == self.h):
            self.shape = "SQUARE"
        else:
            self.shape = "RECT"

    def get_center_pixel(self) -> tuple:
        # We can just use the radius here to find the middle of the image
        return (self.radius_x, self.radius_y)


class Draw:
    """
    Wrapper class for drawing mehanisms and related utilities.

    Args:
        parent (class): The composition parent.

    """

    def __init__(self, parent):
        self.parent = parent

    def opaque_region(
        self,
        base_frame: np.ndarray,
        draw_region: tuple[slice, slice],
        base_weight: float,
        draw_weight: float,
    ) -> np.ndarray:
        yrange, xrange = draw_region
        subregion = base_frame[yrange, xrange]
        white_rect = np.ones(subregion.shape, dtype=np.uint8) * 255
        res = cv.addWeighted(subregion, base_weight, white_rect, draw_weight, 1.0)

        base_frame[yrange, xrange] = res
        return base_frame

    def active_quadrant(
        self, base_frame: np.ndarray, base_weight: float, draw_weight: float
    ) -> np.ndarray:
        marked_quadrant = self.opaque_region(
            base_frame, self.parent.quadrant_subsection, base_weight, draw_weight
        )
        return marked_quadrant

    def bounding_box(
        self,
        base_frame: np.ndarray,
        box_center: tuple[int, int],
        box_size: int,
        base_weight: float,
        draw_weight: float,
    ) -> np.ndarray:
        yrange = slice(box_center[1] - box_size, box_center[1] + box_size)
        xrange = slice(box_center[0] - box_size, box_center[0] + box_size)
        new_frame = self.opaque_region(
            base_frame, (yrange, xrange), base_weight, draw_weight
        )
        return new_frame

    def processing_results(
        self, frame: np.ndarray, region: tuple[slice, slice], value: np.ndarray
    ) -> np.ndarray:
        frame[region] = value
        return frame

    def border_around_region(self, image: np.ndarray, thickness: int, color: list[int]):
        h, w = image.shape[:2]
        cv.rectangle(
            image,
            (0, 0),
            (w - 1, h - 1),
            color,
            thickness=thickness,
        )
        return image
