import cv2 as cv
from . import utils
import numpy as np
from . import calculate_rpm as crpm
from .feed import feed
import math
from collections import deque


class BoundingBox:
    """
    Wrapper class for bounding box sub-regions.
    Helps with instantiation of regions.

    Args:
        center (tuple): Coordinate specifying the centerpoint of the bounding box.
        size: (int): specifies the "radius" of the box.
        region (slice:slice): specifies a region (yslice, xslice) as an alternative to center coordinate + size

    """

    def __init__(self, center, size, region, frame_buffer_size, id):
        self.center = center
        self.size = size

        #  Region in OpenCV crop format
        self.region = region
        #  side length =/= size
        self.side_length = self.size * 2
        self.draw = feed.Draw(self)
        self.fb = FrameBuffer(self, frame_buffer_size)
        self.id = id
        self.rank = 0

    def dilate_and_erode(
        self,
        frame: np.ndarray,
        kernel_size: tuple[int, int],
        dil_it: int,
        er_it: int,
    ) -> np.ndarray:
        subregion = frame[self.region]
        kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
        dilated = cv.dilate(subregion, kernel, iterations=dil_it)
        processed_subregion = cv.erode(dilated, kernel, iterations=er_it)
        return processed_subregion

    def area(self):
        return self.side_length * self.side_length

    @classmethod
    def from_center_and_size(cls, center, size, frame_buffer_size, id):
        region = cls.region_from_center_and_size(center, size)
        return cls(center, size, region, frame_buffer_size, id)

    @classmethod
    def from_region(cls, region, frame_buffer_size, id):
        center, size = cls.center_and_size_from_region(region)
        return cls(center, size, region, frame_buffer_size, id)

    @staticmethod
    def region_from_center_and_size(
        center: tuple[int, int], size: int
    ) -> tuple[slice, slice]:
        yrange = slice(center[1] - size, center[1] + size)
        xrange = slice(center[0] - size, center[0] + size)
        return (yrange, xrange)

    @staticmethod
    def center_and_size_from_region(
        region: tuple[slice, slice],
    ) -> tuple[tuple[int, int], int]:
        yrange = region[0]
        xrange = region[1]
        sizey = (yrange.stop - yrange.start) // 2
        sizex = (xrange.stop - xrange.start) // 2
        assert sizey == sizex  # Force square size
        center = ((xrange.start + sizex), (yrange.start + sizey))
        return (center, sizex)


class FrameBuffer:
    """
    A wrapper class for initialized deques. Contains methods that simplify the trivial stuff.

    Args:
        parent (class): The composition parent.

    """

    def __init__(self, parent, size):
        self.parent = parent
        self.average_delta = 0
        self.entries = deque(maxlen=size)

    def insert(self, region: np.ndarray) -> None:
        # Store the processed regions and nice-to-haves in the buffer
        intensity = np.mean(region)

        if len(self.entries) > 0:
            prev_frame_intensity = self.entries[-1]["intensity"]
            intensity_delta = intensity - prev_frame_intensity
        else:
            #  Setting these to 0 reduce startup spikes
            intensity_delta = 0

        entry = {
            "subregion": region,
            "intensity": intensity,
            "intensity_delta": intensity_delta,
        }
        self.entries.append(entry)

    # Only takes the last updated value and updates avgs
    # Designed this way so a user can conditionally update
    def update_color_delta_average(self) -> None:
        vals = []
        for entry in self.entries:
            vals.append(entry["intensity_delta"])
        self.average_delta = np.mean(vals)


class BpmCascade(feed.RpmFromFeed):
    """
    The main class. Contains and/or utilizes the other classes in some way or another.
    "Cascades" bounding boxes; meaning that it contains logic for repeating N boxes
    as separate detection regions. Contains logic for managing these regions.

    Args:
        **kwargs (dict from JSON-config file): see software/config/config_template.json.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.center_of_frame = self.get_center_pixel()
        self.corner = self._get_quadrant_corner_pixel()
        self.hypotenuse_length = self._get_hypotenuse_length()
        self.quadrant_subsection = self._get_quadrant_subsection_slice()
        self.quadrant_axis_map = self._generate_axis_mapping()
        self.draw = feed.Draw(self)
        self.detection_enable_toggle = True
        self.color_delta_update_frequency: int
        self.quadrant: int
        self.all_fb_delta_average = 0
        self.frame_buffer_size: int
        self.stack_boxes_vertically: bool
        self.stack_boxes_horizontally: bool
        self.trim_last_n_boxes: int
        self.start_from_box: int
        self.threshold_multiplier: int
        self.detection_warmup_frames: int
        self.rpm_acceleration_bound: int
        self.turbine_diameter: float

        if self.contrast_multiplier == 1:
            self.adjust_contrast = False
        else:
            self.adjust_contrast = True

    def _generate_axis_mapping(self) -> tuple[int, int]:
        axes = (1, -1)
        if self.quadrant == 2:
            axes = (-1, -1)
        elif self.quadrant == 3:
            axes = (-1, 1)
        elif self.quadrant == 4:
            axes = (1, 1)
        return axes

    # Uses mathematical quadrants, not OpenCV indexing
    def _get_quadrant_corner_pixel(self) -> tuple[int, int]:
        list_index = self.quadrant - 1
        all_corners = [
            (self.w - 1, 0),  # Top right
            (0, 0),  # Top left
            (0, self.h - 1),  # Bottom left
            (self.w - 1, self.h - 1),  # Bottom right
        ]

        corner_pixel = all_corners[list_index]
        return corner_pixel

    def calculate_rpm(self, frame_time: int, fps: float) -> float:
        return crpm.calculate_rpm_from_frame_time(frame_time, fps)

    def update_global_fb_average(self):
        self.rank_and_weight_bounding_boxes()
        sum = []
        for box in self.bounds.values():
            sum.append(box.fb.average_delta)
        self.all_fb_delta_average = np.mean(sum)

    def print_useful_stats(
        self,
        out: deque = deque(maxlen=5),
        frame_ticks: deque = deque(maxlen=1),
        detection_enable_toggle: bool = True,
        threshold: float = 0,
        mode: float = 0,
    ) -> None:
        # Frame counter
        print(
            f"{utils.bcolors.HEADER}Frame: {self.frame_cnt}{utils.bcolors.ENDC} - ",
            end="",
        )

        # Delta, thresholds and mode
        print(
            f"Delta / Threshold / mode: {utils.bcolors.OKCYAN}{utils.bcolors.UNDERLINE}{round(self.all_fb_delta_average, 2)} / {round(threshold, 2)} / {round(mode, 1)}{utils.bcolors.ENDC} - ",
            end="",
        )

        # Detect enabled status
        print(
            f"{'Detect Enabled' if detection_enable_toggle else 'Detect Disabled'} - ",
            end="",
        )

        # RPM data
        print(
            f"RPM: {utils.bcolors.FAIL}{utils.bcolors.BOLD}{0 if not out else round(out[-1], 3)}{utils.bcolors.ENDC} - ",
            end="",
        )

        # Time since last detection
        print(
            f"Last detection {self.frame_cnt - (0 if not frame_ticks else frame_ticks[-1])} frames ago - ",
            end="",
        )

        # Error rate
        print(
            f"Error: {utils.bcolors.FAIL}{utils.bcolors.BOLD}{round(utils.calculate_error_percentage(float((0 if not out else out[-1])), self.real_rpm), 2)}%{utils.bcolors.ENDC}"
        )

    def _get_quadrant_subsection_slice(self) -> tuple[slice, slice]:
        #  This is hardcoded and absolutely awful. I'm sorry
        #  The slice bounds are absolute, while the quadrants
        #  are relative to the center.
        if (self.quadrant == 2) or (self.quadrant == 3):
            xrange = slice(self.corner[0], self.center_of_frame[0])
        else:
            xrange = slice(self.center_of_frame[0], self.corner[0])

        if (self.quadrant == 1) or (self.quadrant == 2):
            yrange = slice(self.corner[1], self.center_of_frame[1])
        else:
            yrange = slice(self.center_of_frame[1], self.corner[1])

        return (yrange, xrange)

    def _get_hypotenuse_length(self) -> float:
        xlen = abs(self.center_of_frame[0] - self.corner[0])
        ylen = abs(self.center_of_frame[1] - self.corner[1])
        hyp_length = math.sqrt((ylen**2) + (xlen**2))
        return hyp_length

    def update_detection_enable_toggle(
        self, intensity_delta, threshold, mode, frame_ticks
    ):
        if (mode - threshold < intensity_delta < mode + threshold) and (
            self.frame_cnt - (0 if not frame_ticks else frame_ticks[-1]) > 10
        ):
            self.detection_enable_toggle = True

    def blade_detection_in_box_regions(self, deviation: float, mode: float) -> bool:
        if self.frame_cnt < getattr(self, "detection_warmup_frames", 0):
            return False

        # It might seem weird to use the all_fb_delta_average here, but in this situation we
        # have already factored in the weighting scheme. So this value already prioritizes boxes.
        if (
            self.all_fb_delta_average > (mode + self.threshold_multiplier * deviation)
            and self.detection_enable_toggle
        ):
            return True
        else:
            return False

    def rank_and_weight_bounding_boxes(self):
        # Get all detection strengths
        buffer_values = {}
        for box in self.bounds.values():
            buffer_values[f"{box.id}"] = box.fb.average_delta

        # Sort boxes based on detection strength
        self.sorted_ids_by_strength = sorted(
            buffer_values, key=buffer_values.__getitem__, reverse=True
        )

        # Apply ranking based on detection strength
        weights = np.linspace(1, 0, len(self.sorted_ids_by_strength))
        for rank, id in enumerate(self.sorted_ids_by_strength):
            self.bounds[id].rank = rank
            self.bounds[id].fb.average_delta = (
                self.bounds[id].fb.average_delta * weights[rank]
            )  # Rank is also just an index

    def boxes_in_radius(self, box_size: int) -> int:
        # In the horizontal or vertical stacking cases,
        # using the radius to the middle of a box's side
        # is better

        box_diameter = 2 * box_size
        if self.stack_boxes_vertically:
            num_boxes = math.floor(self.radius_y / box_diameter)
            return num_boxes
        elif self.stack_boxes_horizontally:
            num_boxes = math.floor(self.radius_x / box_diameter)
            return num_boxes

        # Diagonal case
        else:
            box_diagonal = round(2 * box_size * math.sqrt(2))
            num_boxes = math.floor(self.hypotenuse_length / box_diagonal)
            return num_boxes

    def get_fitted_box_params_from_cfg(self):
        self.target_num_boxes: int
        self.target_box_size: int
        self.resize_boxes: bool
        self.adjust_num_boxes: bool

        return self.fit_box_parameters_to_radius(
            self.target_num_boxes,
            self.target_box_size,
            self.resize_boxes,
            self.adjust_num_boxes,
        )

    def get_dilation_erosion_params(self):
        self.erosion_dilation_kernel_size: list[int]
        self.dilation_iterations: int
        self.erosion_iterations: int

        return (
            (
                self.erosion_dilation_kernel_size[0],
                self.erosion_dilation_kernel_size[1],
            ),
            self.dilation_iterations,
            self.erosion_iterations,
        )

    def fit_box_parameters_to_radius(
        self,
        wanted_num_boxes: int,
        wanted_box_size: int,
        resize_boxes: bool = False,
        adjust_num_boxes: bool = False,
    ) -> tuple[int, int]:
        # Calculate how many boxes can fit with the current box size:
        initial_box_limit = self.boxes_in_radius(wanted_box_size)

        # Initialize as an ideal request
        result_boxes = wanted_num_boxes
        result_size = wanted_box_size

        # Case 1: The current request goes out of bounds (too many boxes for the size).
        if wanted_num_boxes > initial_box_limit:
            if adjust_num_boxes:
                # Instead of resizing, adjust the box count to the maximum available.
                result_boxes = initial_box_limit

            elif resize_boxes:
                # Reduce the box size until it can hold the desired number of boxes.
                while True:
                    if self.boxes_in_radius(result_size) < wanted_num_boxes:
                        result_size -= 1
                    else:
                        break

        # Case 2: The user is within bounds.
        else:
            if resize_boxes:
                # Expand box size until increasing it further would mean we are out of bounds
                while True:
                    if self.boxes_in_radius(result_size + 1) <= initial_box_limit:
                        result_size += 1
                    else:
                        break

            elif adjust_num_boxes:
                # We are within bounds, and we can just set the number of boxes to be the max
                result_boxes = initial_box_limit

        return (result_boxes, result_size)

    def process_rpm_bounds(self):
        # If this is the case then we have a custom diameter we need to calculate the bounds for
        if self.turbine_diameter != 0:
            self.max_rpm = self.calculate_max_rpm_limit(direct_drive=self.direct_drive)

        else:
            # TODO: make this dynamic
            self.max_rpm = 30

    def calculate_max_rpm_limit(self, direct_drive=True):
        # Based on a fitting regression from data points, see thesis
        if direct_drive:
            return (59974.7617 / (self.turbine_diameter**2)) + 8.36949
        else:
            return (911.43187 / (self.turbine_diameter)) + 7.57917

    def rpm_within_bounds(self, rpm, prev_rpm):
        if rpm < self.max_rpm or (
            (prev_rpm - self.rpm_acceleration_bound)
            < rpm
            < (prev_rpm + self.rpm_acceleration_bound)
        ):
            return True
        else:
            return False

    def cascade_bounding_boxes(
        self,
        num_boxes: int,
        box_size,
    ) -> dict[str, BoundingBox]:
        bounds = {}

        #  TODO: figure out a smarter way to do this axis stuff
        delta_y = (
            0
            if self.stack_boxes_horizontally
            else box_size * (2 * self.quadrant_axis_map[1])
        )
        delta_x = (
            0
            if self.stack_boxes_vertically
            else box_size * (2 * self.quadrant_axis_map[0])
        )

        offset_y = (
            self.corner[1]
            - self.center_of_frame[1]
            + (box_size * self.quadrant_axis_map[1] - 1)
        )
        offset_x = (
            self.corner[0]
            - self.center_of_frame[0]
            + (box_size * self.quadrant_axis_map[0])
        )

        box_range = slice(self.start_from_box - 1, num_boxes - self.trim_last_n_boxes)

        # Cascade boxes
        for i in range(box_range.start, box_range.stop):
            box_x = round(offset_x + delta_x * i)
            box_y = round(offset_y + delta_y * i)
            box_center = (box_x, box_y)

            bounds[f"{i}"] = BoundingBox.from_center_and_size(
                box_center, box_size, self.frame_buffer_size, i
            )
        self.bounds = bounds
        return bounds
