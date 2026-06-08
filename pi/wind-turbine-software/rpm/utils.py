import numpy as np
from collections import deque
import json


def calculate_error_percentage(
    measured_value: float | None, actual_value: float | None
) -> float | None:
    if actual_value is None:
        return None

    if measured_value is not None and actual_value != 0:
        error_percentage = abs(measured_value - actual_value) / actual_value * 100
    else:
        error_percentage = np.nan
    return error_percentage


def print_statistics(
    rpms: list,
    errors: list,
    real_rpm: float | None = None,
    rounding_factor: int = 2,
    verbose: bool = False,
) -> None:
    avg_rpm = float(round(np.average(rpms), rounding_factor))

    if real_rpm is not None:
        avg_error = round(np.average(errors), rounding_factor)
        avg_error_from_real = calculate_error_percentage(avg_rpm, real_rpm)

        if verbose:
            for index, _ in enumerate(rpms):
                print(
                    "RPM:",
                    round(rpms[index], rounding_factor),
                    f"Error: {round(errors[index], rounding_factor)}%",
                )

        print(f"Average rpm: {avg_rpm}")
        print(f"Average RPM error percentage from real RPM: {avg_error_from_real}%")
        print(f"Average of all error percentages: {avg_error}%")

    else:
        avg_error = None
        avg_error_from_real = None
        if verbose:
            for index, _ in enumerate(rpms):
                print("RPM:", round(rpms[index], rounding_factor))

        print(f"Average rpm: {avg_rpm}")

    return None


def parse_json(file_path: str) -> dict:
    with open(file_path) as config_file:
        params = json.load(config_file)
        return params


def dynamic_log_string(
    rpm_monitor,
    tick_timestamp,
    colorvals,
    rpm_buffer: deque | list,
    print_error=False,
    real_rpm=0,
):
    frame_tick_printstr = (
        str(rpm_monitor.frame_cnt) if rpm_monitor.log_frame_ticks else None
    )
    tick_timestamp_printstr = (
        str(tick_timestamp) if rpm_monitor.log_timestamps else None
    )

    # Sorry
    color_printstr = (
        "".join(
            (str(color) + ("" if color == colorvals[-1] else "/"))
            for color in colorvals
        )
        if rpm_monitor.log_color_values
        else None
    )

    # This one is not negotiable and the thought of making this removable is silly
    rpm_printstr = str(np.mean(rpm_buffer))

    print_items = [
        frame_tick_printstr,
        tick_timestamp_printstr,
        color_printstr,
    ]

    total_out_string = ""
    for printstr in print_items:
        if printstr is not None:
            total_out_string += printstr + ","

    # The final addition here is a bit weird so we do that one separately
    total_out_string += rpm_printstr

    if print_error:
        error = calculate_error_percentage(float(rpm_printstr), real_rpm)
        total_out_string += "," + str(error)

    total_out_string += "\n"

    return total_out_string


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def write_output(
    turbine_id: int, frame_number: int, rpm: list[float], real_rpm: int | None = None
):
    with open(f"runs/run_turbine{str(turbine_id)}.csv", "a") as output_file:
        output_file.write(
            f"{frame_number}, {rpm}, {calculate_error_percentage(float(np.average(rpm)), real_rpm)}\n"
        )


def find_top_n_modes(
    data: list | deque, n: int = 1, return_counts=False, mode_round_delta_to_digit=1
) -> list:
    arr = np.asarray([round(value, mode_round_delta_to_digit) for value in data])

    values, counts = np.unique(arr, return_counts=True)

    sorted_indices = np.argsort(-counts)
    top_values = values[sorted_indices][:n]
    top_counts = counts[sorted_indices][:n]

    if return_counts:
        # Return as a list of (value, frequency) pairs
        return list(zip(top_values, top_counts))
    else:
        return list(top_values)
