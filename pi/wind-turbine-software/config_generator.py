import sys
from rpm import bpm_cascade
import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow,
    QApplication,
    QLabel,
    QFileDialog,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QWidget,
    QSizePolicy,
    QButtonGroup,
    QGroupBox,
    QFormLayout,
    QGridLayout,
    QStackedWidget,
    QDialog,
    QDialogButtonBox,
    QCheckBox,
)
from PyQt6.QtGui import QIcon, QPixmap, QImage
import cv2


class CustomDialog(QDialog):
    def __init__(self, parent, success=False):
        super().__init__()
        self.parent = parent

        QBtn = (
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        if success:
            self.setWindowTitle("Config generation successful!")
            self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
            self.buttonBox.button(QDialogButtonBox.StandardButton.Cancel).setText(
                "Exit"
            )
            message = QLabel(
                f"Config generation was successful and the file written to {self.parent.file_loc}."
            )
            message2 = QLabel(
                'Use run_wind_turbine.py with this config when you want to test it.'
            )
        else:
            self.setWindowTitle("Config generation failed!")
            message = QLabel(
                "Config file generation was unsuccessful. The config file may not have been generated."
            )
            message2 = QLabel("Press OK to close this dialog.")
            self.buttonBox.button(QDialogButtonBox.StandardButton.Ok).setText("OK")
            self.buttonBox.button(QDialogButtonBox.StandardButton.Cancel).setText(
                "Exit"
            )

        layout = QVBoxLayout()
        layout.addWidget(message)
        layout.addWidget(message2)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)


class PreviewWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setLayout(QVBoxLayout())
        self.xrange = None
        self.yrange = None
        self.init_image_preview()

    def init_image_preview(self):
        self.image_preview = QLabel()
        self.image_preview.setScaledContents(True)
        self.image_preview.setObjectName("image_preview")
        self.image_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.update_image_preview(None, None)

        # Add label to your panel’s layout
        self.layout().addWidget(self.image_preview, stretch=1)

    def update_image_preview(self, xrange=None, yrange=None, draw_boxes=True):
        self.xrange = xrange
        self.yrange = yrange
        if not self.parent.video_path:
            self.image_preview.setText("No video selected")
            return

        self.video = cv2.VideoCapture(self.parent.video_path)
        ret, frame = self.video.read()
        self.video.release()

        if ret:
            if draw_boxes:
                frame = self.parent.draw_preview_boxes(frame)
            elif self.xrange is not None and self.yrange is not None:
                frame = frame[self.yrange, self.xrange]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_qimage = convert_cvimg_to_qimg(frame)
            frame_pixmap = QPixmap(frame_qimage)
            self.image_preview.setPixmap(frame_pixmap)
            self.adjustSize()
        else:
            self.image_preview.setText("No video selected")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        # Window setup
        self.setWindowTitle("Config Generator")
        self.setGeometry(400, 000, 700, 500)
        self.setWindowIcon(QIcon("assets/gui_icon.png"))
        self.video_path = None
        self.apply_global_styles()

        # Top level layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        # Feed preview
        self.preview = PreviewWindow(self)

        # Config generator panel
        self.control_panel = QWidget()
        self.control_panel.setLayout(QVBoxLayout())

        # The mode-dependent panel with parameters
        self.bpm_mode_config_panel = QWidget()
        self.bpm_mode_config_panel_layout = QVBoxLayout(self.bpm_mode_config_panel)

        self.opticalflow_mode_config_panel = QWidget()
        self.opticalflow_mode_config_panel_layout = QVBoxLayout(
            self.opticalflow_mode_config_panel
        )
        self.opticalflow_mode_config_panel.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        )
        self.update_panel_max_height()

        self.mode_stack_manager = QStackedWidget()
        self.mode_stack_manager.addWidget(self.bpm_mode_config_panel)
        self.mode_stack_manager.addWidget(self.opticalflow_mode_config_panel)

        # Main params
        self.initln_path_select()
        self.initln_crop_points()
        self.initln_feed_details()
        self.initln_turbine_type()
        #  Mode select
        self.initln_mode_select()
        self.update_mode_dependent_params_view()

        #  BPM: Detection params
        self.initln_box_params()
        self.initln_box_stacking()
        self.initln_detection_params()

        #  BPM: Processing params
        self.initln_kernel_params()

        #  BPM: additional customizations
        self.initln_output_customization()

        #  BPM: Generate config button
        self.initln_generate_config()

        #  Opticalflow: deadzone parameters
        self.initln_deadzone_shape()
        self.initln_deadzone_params()

        #  Opticalflow: constants for calculation
        self.initln_calc_constants()

        # Composition
        self.main_layout.addWidget(self.control_panel, stretch=1)
        self.main_layout.addWidget(self.mode_stack_manager, stretch=1)

        self.preview.show()
        self.main_widget.setLayout(self.main_layout)

    def initln_turbine_type(self):
        self.turbine_type_container = QGroupBox("Wind turbine type")
        self.turbine_type_container_layout = QHBoxLayout(self.turbine_type_container)
        self.turbine_type_group = QButtonGroup(self)
        self.turbine_type_group.setExclusive(True)
        self.turbine_type_container_layout.setSpacing(10)
        self.turbine_type_container_layout.setContentsMargins(0, 5, 0, 5)

        turbine_type_direct_drive = QPushButton(parent=self, text="Direct drive")
        turbine_type_direct_drive.setObjectName("orientation_button")
        turbine_type_direct_drive.setCheckable(True)
        turbine_type_direct_drive.setChecked(True)

        turbine_type_geared_drive = QPushButton(parent=self, text="Geared drive")
        turbine_type_geared_drive.setObjectName("orientation_button")
        turbine_type_geared_drive.setCheckable(True)

        self.turbine_type_group.addButton(turbine_type_direct_drive)
        self.turbine_type_group.addButton(turbine_type_geared_drive)

        self.turbine_type_container_layout.addWidget(turbine_type_direct_drive)
        self.turbine_type_container_layout.addWidget(turbine_type_geared_drive)

        self.control_panel.layout().addWidget(self.turbine_type_container)

    def initln_path_select(self):
        self.path_select = QGroupBox("Path selection")
        self.path_select_layout = QFormLayout(self.path_select)
        self.path_select_layout.setSpacing(10)
        self.path_select_layout.setContentsMargins(0, 5, 0, 5)

        path_select_input, self.fld_path_select, _ = self.create_labeled_field(
            "Feed path", objname="input_field"
        )

        self.btn_path_select = QPushButton(parent=self, text="Browse..")
        self.btn_path_select.setObjectName("browse_button")
        self.btn_path_select.clicked.connect(self.dialog_path_select)
        self.btn_path_select.clicked.connect(self.preview.update_image_preview)

        path_select_input.layout().addWidget(self.btn_path_select)
        self.path_select_layout.addRow(path_select_input)
        self.control_panel.layout().addWidget(self.path_select)

    def initln_deadzone_shape(self):
        shapeselect = QGroupBox("Deadzone shape selection")
        shapeselect_layout = QHBoxLayout(shapeselect)
        shapeselect_layout.setSpacing(10)
        shapeselect_layout.setContentsMargins(0, 5, 0, 5)

        self.shape_group = QButtonGroup()

        circle_mode = QPushButton(parent=self, text="Circle")
        circle_mode.setCheckable(True)
        circle_mode.setAutoExclusive(True)
        circle_mode.setChecked(True)
        circle_mode.setObjectName("orientation_button")

        square_mode = QPushButton(parent=self, text="Square")
        square_mode.setCheckable(True)
        square_mode.setAutoExclusive(True)
        square_mode.setObjectName("orientation_button")

        self.shape_group.addButton(circle_mode)
        self.shape_group.addButton(square_mode)

        shapeselect_layout.addWidget(circle_mode)
        shapeselect_layout.addWidget(square_mode)

        self.opticalflow_mode_config_panel_layout.addWidget(shapeselect)

    def initln_output_customization(self):
        self.output_params = QGroupBox("Extra configuration details")
        self.output_params_layout = QGridLayout(self.output_params)
        self.output_params_layout.setSpacing(10)
        self.output_params_layout.setContentsMargins(0, 0, 0, 0)

        rpm_buffer, rpm_buffer_fld, rpm_buffer_lbl = self.create_labeled_field(
            "RPM smoothing", placeholder_text="15", objname="input_field"
        )
        rpm_bound, rpm_bound_fld, rpm_bound_lbl = self.create_labeled_field(
            "RPM acceleration bound", placeholder_text="3", objname="input_field"
        )
        rpm_zero_timeout, rpm_zero_timeout_fld, rpm_zero_timeout_lbl = (
            self.create_labeled_field(
                "RPM zero timeout seconds", placeholder_text="10", objname="input_field"
            )
        )
        turbine_diameter, turbine_diameter_fld, turbine_diameter_lbl = (
            self.create_labeled_field(
                "Turbine diameter", placeholder_text="0", objname="input_field"
            )
        )

        self.timestamp_check = QCheckBox("Log timestamp of detection")
        self.frame_tick_check = QCheckBox("Log frame number of detection")
        self.color_value_check = QCheckBox("Log measured color values")
        self.timestamp_check.setChecked(True)

        for checkbox in [
            self.timestamp_check,
            self.frame_tick_check,
            self.color_value_check,
        ]:
            checkbox.setMinimumSize(150, 30)

        self.output_params_layout.addWidget(rpm_buffer_lbl, 0, 0)
        self.output_params_layout.addWidget(rpm_buffer_fld, 0, 1)
        self.output_params_layout.addWidget(rpm_bound_lbl, 1, 0)
        self.output_params_layout.addWidget(rpm_bound_fld, 1, 1)
        self.output_params_layout.addWidget(rpm_zero_timeout_lbl, 2, 0)
        self.output_params_layout.addWidget(rpm_zero_timeout_fld, 2, 1)
        self.output_params_layout.addWidget(turbine_diameter_lbl, 3, 0)
        self.output_params_layout.addWidget(turbine_diameter_fld, 3, 1)

        self.output_params_layout.addWidget(self.timestamp_check, 0, 2)
        self.output_params_layout.addWidget(self.frame_tick_check, 1, 2)
        self.output_params_layout.addWidget(self.color_value_check, 2, 2)
        self.bpm_mode_config_panel_layout.addWidget(self.output_params)

    def initln_calc_constants(self):
        self.calc_params = QGroupBox("Flow vector RPM calculation constants")
        self.calc_params_layout = QGridLayout(self.calc_params)
        self.calc_params_layout.setSpacing(10)
        self.calc_params_layout.setContentsMargins(0, 5, 0, 5)

        ga, ga_fld, ga_lbl = self.create_labeled_field(
            "Ground angle", placeholder_text="0.7", objname="input_field"
        )
        pt, pt_fld, pt_lbl = self.create_labeled_field(
            "Pixel tracking threshold", placeholder_text="10", objname="input_field"
        )

        self.calc_params_layout.addWidget(ga_lbl, 0, 0)
        self.calc_params_layout.addWidget(ga_fld, 0, 1)
        self.calc_params_layout.addWidget(pt_lbl, 0, 2)
        self.calc_params_layout.addWidget(pt_fld, 0, 3)

        self.opticalflow_mode_config_panel_layout.addWidget(self.calc_params)

    def initln_deadzone_params(self):
        self.deadzone_params = QGroupBox("Dead zone parameters")
        self.deadzone_layout = QGridLayout(self.deadzone_params)
        self.deadzone_layout.setSpacing(10)
        self.deadzone_layout.setContentsMargins(0, 5, 0, 5)

        dzx, dzx_fld, dzx_lbl = self.create_labeled_field(
            "Deadzone size x", placeholder_text="10", objname="input_field"
        )
        dzy, dzy_fld, dzy_lbl = self.create_labeled_field(
            "Deadzone size y", placeholder_text="10", objname="input_field"
        )
        dzoffx, dzoffx_fld, dzoffx_lbl = self.create_labeled_field(
            "Deadzone offset x", placeholder_text="0", objname="input_field"
        )
        dzoffy, dzoffy_fld, dzoffy_lbl = self.create_labeled_field(
            "Deadzone offset y", placeholder_text="0", objname="input_field"
        )

        self.deadzone_layout.addWidget(dzx_lbl, 0, 0)
        self.deadzone_layout.addWidget(dzx_fld, 0, 1)
        self.deadzone_layout.addWidget(dzy_lbl, 0, 2)
        self.deadzone_layout.addWidget(dzy_fld, 0, 3)
        self.deadzone_layout.addWidget(dzoffx_lbl, 1, 0)
        self.deadzone_layout.addWidget(dzoffx_fld, 1, 1)
        self.deadzone_layout.addWidget(dzoffy_lbl, 1, 2)
        self.deadzone_layout.addWidget(dzoffy_fld, 1, 3)

        self.opticalflow_mode_config_panel_layout.addWidget(self.deadzone_params)

    def initln_kernel_params(self):
        self.kernel_params = QGroupBox("Image region processing parameters")
        self.kernel_params_layout = QGridLayout(self.kernel_params)
        self.kernel_params_layout.setSpacing(10)
        self.kernel_params_layout.setContentsMargins(0, 5, 0, 5)

        kernel_size, kernel_size_fld, kernel_size_lbl = self.create_labeled_field(
            "Kernel size", placeholder_text="10", objname="input_field"
        )
        dilation_iterations, dilation_iterations_fld, dilation_iterations_lbl = (
            self.create_labeled_field(
                "Dilation iterations", placeholder_text="2", objname="input_field"
            )
        )
        erosion_iterations, erosion_iterations_fld, erosion_iterations_lbl = (
            self.create_labeled_field(
                "Erosion iterations", placeholder_text="1", objname="input_field"
            )
        )

        self.kernel_params_layout.addWidget(kernel_size_lbl, 0, 0)
        self.kernel_params_layout.addWidget(kernel_size_fld, 0, 1)
        self.kernel_params_layout.addWidget(dilation_iterations_lbl, 0, 2)
        self.kernel_params_layout.addWidget(dilation_iterations_fld, 0, 3)
        self.kernel_params_layout.addWidget(erosion_iterations_lbl, 0, 4)
        self.kernel_params_layout.addWidget(erosion_iterations_fld, 0, 5)

        self.bpm_mode_config_panel_layout.addWidget(self.kernel_params)

    def initln_box_stacking(self):
        self.box_stack_container = QGroupBox("Box stacking orientation")
        self.box_stack_container_layout = QHBoxLayout(self.box_stack_container)
        self.stack_group = QButtonGroup(self)
        self.stack_group.setExclusive(True)
        self.box_stack_container_layout.setSpacing(10)
        self.box_stack_container_layout.setContentsMargins(0, 5, 0, 5)

        stack_mode_hor = QPushButton(parent=self, text="Horizontal")
        stack_mode_hor.setObjectName("orientation_button")
        stack_mode_hor.setCheckable(True)
        stack_mode_hor.setChecked(True)

        stack_mode_vert = QPushButton(parent=self, text="Vertical")
        stack_mode_hor.setObjectName("orientation_button")
        stack_mode_vert.setCheckable(True)

        stack_mode_diag = QPushButton(parent=self, text="Diagonal")
        stack_mode_hor.setObjectName("orientation_button")
        stack_mode_diag.setCheckable(True)

        self.stack_group.addButton(stack_mode_hor)
        self.stack_group.addButton(stack_mode_vert)
        self.stack_group.addButton(stack_mode_diag)

        self.box_stack_container_layout.addWidget(stack_mode_hor)
        self.box_stack_container_layout.addWidget(stack_mode_vert)
        self.box_stack_container_layout.addWidget(stack_mode_diag)

        self.bpm_mode_config_panel_layout.addWidget(self.box_stack_container)

    def initln_box_params(self):
        self.box_params = QGroupBox("Detection zone configuration")
        box_params_layout = QGridLayout(self.box_params)
        box_params_layout.setSpacing(10)
        box_params_layout.setContentsMargins(0, 5, 0, 5)

        quadrant_container, quadrant_fld, quadrant_label = self.create_labeled_field(
            "Quadrant", placeholder_text="1", objname="input_field"
        )
        num_boxes, num_boxes_fld, num_boxes_label = self.create_labeled_field(
            "Number of boxes", placeholder_text="10", objname="input_field"
        )
        box_size, box_size_fld, box_size_label = self.create_labeled_field(
            "Box size", placeholder_text="10"
        )
        start_from_box, start_from_box_fld, start_from_box_label = (
            self.create_labeled_field(
                "Start from box", placeholder_text="0", objname="input_field"
            )
        )
        trim_last_n, trim_last_n_fld, trim_last_n_label = self.create_labeled_field(
            "Trim last N boxes", placeholder_text="0", objname="input_field"
        )

        self.box_preview = QPushButton(parent=self, text="Preview")
        self.box_preview.setObjectName("previewbutton")

        box_params_layout.addWidget(quadrant_label, 0, 0)
        box_params_layout.addWidget(quadrant_fld, 0, 1)
        box_params_layout.addWidget(num_boxes_label, 0, 2)
        box_params_layout.addWidget(num_boxes_fld, 0, 3)
        box_params_layout.addWidget(box_size_label, 0, 4)
        box_params_layout.addWidget(box_size_fld, 0, 5)
        box_params_layout.addWidget(start_from_box_label, 1, 0)
        box_params_layout.addWidget(start_from_box_fld, 1, 1)
        box_params_layout.addWidget(trim_last_n_label, 1, 2)
        box_params_layout.addWidget(trim_last_n_fld, 1, 3)
        box_params_layout.addWidget(self.box_preview, 1, 5)

        self.box_preview.clicked.connect(self.preview_box_params)

        self.bpm_mode_config_panel_layout.addWidget(self.box_params)

    def preview_box_params(self):
        if not self.video_path:
            self.preview.image_preview.setText("Select a video before previewing boxes")
            return

        self.update_crop_points()
        self.preview.update_image_preview(
            xrange=self.xrange,
            yrange=self.yrange,
            draw_boxes=True,
        )

    def build_preview_params(self):
        params = self.json_sanitize(self.extract_params())
        params["target"] = self.video_path
        return params

    def draw_preview_boxes(self, frame):
        params = self.build_preview_params()
        temp_feed = bpm_cascade.BpmCascade(**params)
        preview = temp_feed.get_frame()
        if preview is None:
            return frame

        display_frame = preview.copy()
        box_params = temp_feed.get_fitted_box_params_from_cfg()
        bounds = temp_feed.cascade_bounding_boxes(*box_params)
        kernel_er_dil_params = temp_feed.get_dilation_erosion_params()

        for bounding_box in bounds.values():
            processed_region = bounding_box.dilate_and_erode(
                preview, *kernel_er_dil_params
            )
            bounding_box.fb.insert(processed_region)
            bounding_box.draw.border_around_region(processed_region, 1, [0, 255, 0])
            display_frame = bounding_box.draw.processing_results(
                display_frame, bounding_box.region, processed_region
            )

        return display_frame

    def initln_detection_params(self):
        self.detection_params = QGroupBox("Detection parameters")
        self.detection_params_layout = QGridLayout(self.detection_params)
        self.detection_params_layout.setSpacing(10)
        self.detection_params_layout.setContentsMargins(0, 5, 0, 5)

        fb_size_container, fb_fld, fb_lbl = self.create_labeled_field(
            "Frame buffer size", "5", objname="input_field"
        )

        update_frequency_container, update_freq_fld, update_freq_lbl = (
            self.create_labeled_field("Update frequency", "2", objname="input_field")
        )

        (
            contrast_multiplier_container,
            contrast_multiplier_fld,
            contrast_multiplier_label,
        ) = self.create_labeled_field(
            "Contrast multiplier override", "1", objname="input_field"
        )

        (
            threshold_multiplier_container,
            threshold_multiplier_fld,
            threshold_multiplier_label,
        ) = self.create_labeled_field(
            "Detection threshold multiplier", "1", objname="input_field"
        )
        detection_warmup_frames_container, detection_warmup_frames_fld, detection_warmup_frames_label = (
            self.create_labeled_field(
                "Detection warmup frames", "0", objname="input_field"
            )
        )

        self.detection_params_layout.addWidget(fb_lbl, 0, 0)
        self.detection_params_layout.addWidget(fb_fld, 0, 1)
        self.detection_params_layout.addWidget(update_freq_lbl, 0, 2)
        self.detection_params_layout.addWidget(update_freq_fld, 0, 3)
        self.detection_params_layout.addWidget(contrast_multiplier_label, 1, 0)
        self.detection_params_layout.addWidget(contrast_multiplier_fld, 1, 1)
        self.detection_params_layout.addWidget(threshold_multiplier_label, 1, 2)
        self.detection_params_layout.addWidget(threshold_multiplier_fld, 1, 3)
        self.detection_params_layout.addWidget(detection_warmup_frames_label, 2, 0)
        self.detection_params_layout.addWidget(detection_warmup_frames_fld, 2, 1)

        self.bpm_mode_config_panel_layout.addWidget(self.detection_params)

    def initln_feed_details(self):
        feed_details = QGroupBox("Basic feed parameters")
        feed_details_layout = QFormLayout(feed_details)
        feed_details_layout.setSpacing(10)
        feed_details_layout.setContentsMargins(0, 5, 0, 5)

        id_container, id_field, id_label = self.create_labeled_field(
            "Run ID", placeholder_text="1", objname="input_field"
        )
        fps_container, fps_field, fps_label = self.create_labeled_field(
            "Feed FPS", placeholder_text="60", objname="input_field"
        )
        rpm_container, rpm_field, rpm_label = self.create_labeled_field(
            "Real RPM", placeholder_text="0", objname="input_field"
        )

        feed_details_layout.addRow(id_label, id_field)
        feed_details_layout.addRow(fps_label, fps_field)
        feed_details_layout.addRow(rpm_label, rpm_field)
        self.control_panel.layout().addWidget(feed_details)

    def update_mode_dependent_params_view(self):
        if self.mode_group.checkedButton().text() == "BPM":
            self.mode_stack_manager.setCurrentWidget(self.bpm_mode_config_panel)
        else:
            self.mode_stack_manager.setCurrentWidget(self.opticalflow_mode_config_panel)

    def initln_generate_config(self):
        config_button = QWidget()
        config_button_layout = QHBoxLayout(config_button)
        config_button_layout.setSpacing(10)
        config_button_layout.setContentsMargins(0, 5, 0, 5)
        save_path_container, self.save_path_field, save_path_label = (
            self.create_labeled_field("Save as", placeholder_text="config.json")
        )

        btn = QPushButton(text="Generate config")
        btn.setObjectName("generate_config")
        btn.pressed.connect(self.generate_config)
        config_button_layout.addWidget(save_path_label)
        config_button_layout.addWidget(self.save_path_field)
        config_button_layout.addWidget(btn)
        self.control_panel.layout().addWidget(config_button)

    def generate_config(self):
        json_params = self.extract_params()
        json_params_sanitized = self.json_sanitize(json_params)

        self.file_loc = (
            self.save_path_field.text()
            if self.save_path_field.text() != ""
            else self.save_path_field.placeholderText()
        )
        if not self.file_loc.lower().endswith(".json"):
            self.file_loc += ".json"

        try:
            Path("config").mkdir(parents=True, exist_ok=True)
            with open(Path("config") / self.file_loc, "w", encoding="utf-8") as f:
                json.dump(json_params_sanitized, f, ensure_ascii=False, indent=4)
            result = CustomDialog(self, success=True).exec()
            if result != QDialog.DialogCode.Accepted:
                sys.exit(0)

        except Exception as e:
            print(e)
            CustomDialog(self, success=False).exec()
            sys.exit(1)

    def extract_params(self):
        items = self.findChildren(QLineEdit)
        items_dict = dict()
        for item in items:
            if item.text() != "":
                items_dict[item.property("label")] = item.text()
            else:
                items_dict[item.property("label")] = item.placeholderText()

        stack_method = self.stack_group.checkedButton().text()
        if stack_method == "Horizontal":
            items_dict["stack_boxes_vertically"] = False
            items_dict["stack_boxes_horizontally"] = True
        elif stack_method == "Vertical":
            items_dict["stack_boxes_vertically"] = True
            items_dict["stack_boxes_horizontally"] = False
        elif stack_method == "Diagonal":
            items_dict["stack_boxes_vertically"] = False
            items_dict["stack_boxes_horizontally"] = False

        turbine_type = self.turbine_type_group.checkedButton().text()
        if turbine_type == "Direct drive":
            items_dict["direct_drive"] = True
        else:
            items_dict["direct_drive"] = False

        mode = self.mode_group.checkedButton().text()
        if mode == "BPM":
            items_dict["mode"] = "bpm"
        else:
            items_dict["mode"] = "opticalflow"

        items_dict["log_timestamps"] = (
            "True" if self.timestamp_check.isChecked() else "False"
        )
        items_dict["log_color_values"] = (
            "True" if self.color_value_check.isChecked() else "False"
        )
        items_dict["log_frame_ticks"] = (
            "True" if self.frame_tick_check.isChecked() else "False"
        )

        return items_dict

    def json_sanitize(self, args: dict) -> dict:
        args["crop_points"] = [
            [args["From y"], args["To y"]],
            [args["From x"], args["To x"]],
        ]
        del args["From y"]
        del args["To y"]
        del args["From x"]
        del args["To x"]

        args["deadzone_size"] = [
            int(args["Deadzone size x"]),
            int(args["Deadzone size y"]),
        ]
        del args["Deadzone size x"]
        del args["Deadzone size y"]

        args["deadzone_shape"] = self.shape_group.checkedButton().text().lower()

        remapped_args = {key_map.get(k, k): v for k, v in args.items()}
        remapped_args["source_type"] = "video"
        remapped_args["video_device"] = "/dev/video0"
        remapped_args["width"] = 1920
        remapped_args["height"] = 1080
        remapped_args["raw_path"] = "/tmp/wind_turbine_frame.raw"
        remapped_args["bayer_pattern"] = "BGGR"
        remapped_args["preview_mode"] = "processed"
        remapped_args["black_level"] = 64
        remapped_args["resize_boxes"] = False
        remapped_args["adjust_num_boxes"] = False

        items_to_skip = [
            "source_type",
            "target",
            "video_device",
            "raw_path",
            "bayer_pattern",
            "preview_mode",
            "black_level",
            "stack_boxes_horizontally",
            "stack_boxes_vertically",
            "Save as",
            "mode",
            "deadzone_size",
            "deadzone_shape",
            "direct_drive",
            "resize_boxes",
            "adjust_num_boxes",
        ]
        for item in remapped_args:
            if item in items_to_skip:
                continue
            if item in [
                "contrast_multiplier",
                "threshold_multiplier",
                "real_rpm",
                "ground_angle",
                "fps",
                "rpm_acceleration_bound",
                "rpm_zero_timeout_seconds",
                "turbine_diameter",
            ]:
                remapped_args[item] = float(remapped_args[item])
                continue
            if item == "crop_points":
                for pair in remapped_args[item]:
                    pair[0] = int(pair[0])
                    pair[1] = int(pair[1])
                continue

            if item == "erosion_dilation_kernel_size":
                remapped_args["erosion_dilation_kernel_size"] = [
                    int(float(remapped_args["erosion_dilation_kernel_size"])),
                    int(float(remapped_args["erosion_dilation_kernel_size"])),
                ]
                continue
            if item in ["log_timestamps", "log_color_values", "log_frame_ticks"]:
                remapped_args[item] = True if remapped_args[item] == "True" else False
                continue

            # We should have escaped all the baddies by now, so this is practically an else clause.
            # Remember to "continue" on all other clauses else you'll intify everything
            remapped_args[item] = int(float(remapped_args[item]))

        return remapped_args

    def initln_mode_select(self):
        bar = QGroupBox("RPM calculation mode")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setSpacing(10)
        bar_layout.setContentsMargins(0, 5, 0, 5)

        self.mode_group = QButtonGroup()

        bpm_mode = QPushButton(parent=self, text="BPM")
        bpm_mode.setCheckable(True)
        bpm_mode.setAutoExclusive(True)
        bpm_mode.setChecked(True)
        bpm_mode.setObjectName("mode_button")

        opticalflow_mode = QPushButton(parent=self, text="Optical flow")
        opticalflow_mode.setCheckable(True)
        opticalflow_mode.setAutoExclusive(True)
        opticalflow_mode.setObjectName("mode_button")

        self.mode_group.addButton(bpm_mode)
        self.mode_group.addButton(opticalflow_mode)
        self.mode_group.buttonToggled.connect(self.update_mode_dependent_params_view)

        bar_layout.addWidget(bpm_mode)
        bar_layout.addWidget(opticalflow_mode)

        self.control_panel.layout().addWidget(bar)

    def dialog_path_select(self) -> QFileDialog:
        file_path = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select directory",
            directory=".",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        self.fld_path_select.setText(file_path[0])
        self.video_path = file_path[0]

    def create_labeled_field(self, label_text, placeholder_text="", objname=None):
        container = QWidget()
        container.setLayout(QHBoxLayout())

        label = QLabel(parent=self, text=label_text)
        field = QLineEdit(placeholderText=placeholder_text)
        field.setProperty("label", label_text)

        container.layout().addWidget(label)
        container.layout().addWidget(field)

        if objname is not None:
            field.setObjectName(objname)

        return container, field, label

    def update_crop_points(self):
        try:
            from_x_text = self.fld_from_x.text()
            to_x_text = self.fld_to_x.text()
            from_y_text = self.fld_from_y.text()
            to_y_text = self.fld_to_y.text()

            from_x = int(from_x_text) if from_x_text.strip() else 0
            to_x = int(to_x_text) if to_x_text.strip() else 100
            from_y = int(from_y_text) if from_y_text.strip() else 0
            to_y = int(to_y_text) if to_y_text.strip() else 100

            self.xrange = slice(from_x, to_x)
            self.yrange = slice(from_y, to_y)
            self.preview.update_image_preview(xrange=self.xrange, yrange=self.yrange)

        except ValueError:
            print("Please enter valid integer values for all crop coordinates")

    def closeEvent(self, event):
        # This ensures the child window (preview) closes when parent is closed
        if self.preview:
            self.preview.close()
        event.accept()

    def initln_crop_points(self):
        self.coord_group = QGroupBox("Crop point selection")
        self.coord_layout = QGridLayout(self.coord_group)
        self.coord_layout.setSpacing(10)
        self.coord_layout.setContentsMargins(5, 5, 5, 5)

        from_x, self.fld_from_x, _ = self.create_labeled_field(
            "From x", placeholder_text="0", objname="input_field"
        )
        to_x, self.fld_to_x, _ = self.create_labeled_field(
            "To x", placeholder_text="1920", objname="input_field"
        )
        from_y, self.fld_from_y, _ = self.create_labeled_field(
            "From y", placeholder_text="0", objname="input_field"
        )
        to_y, self.fld_to_y, _ = self.create_labeled_field(
            "To y", placeholder_text="1080", objname="input_field"
        )

        self.refresh_crop = QPushButton(parent=self, text="Refresh")
        self.refresh_crop.setObjectName("refresh_button")
        self.refresh_crop.clicked.connect(self.update_crop_points)

        self.coord_layout.addWidget(from_x, 0, 0)
        self.coord_layout.addWidget(self.fld_from_x, 0, 1)
        self.coord_layout.addWidget(to_x, 0, 2)
        self.coord_layout.addWidget(self.fld_to_x, 0, 3)
        self.coord_layout.addWidget(from_y, 1, 0)
        self.coord_layout.addWidget(self.fld_from_y, 1, 1)
        self.coord_layout.addWidget(to_y, 1, 2)
        self.coord_layout.addWidget(self.fld_to_y, 1, 3)

        self.coord_layout.addWidget(self.refresh_crop, 1, 4)
        self.control_panel.layout().addWidget(self.coord_group)

    def apply_global_styles(self):
        # Apply the stylesheet here
        self.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
            background-color: #f0f0f0;
            font-size: 10pt;
        }

        /* Window title bar styling */
        QMainWindow::title {
            background-color: #333333;
            color: white;
        }

        /* Main container */
        QWidget#main_container {
            background-color: #f0f0f0;
            padding: 15px;
        }

        /* Section headings */
        QLabel.section_heading {
            font-weight: bold;
            color: #333333;
            font-size: 11pt;
            padding-top: 10px;
        }

        /* Group box styling */
        QGroupBox {
            font-weight: bold;
            border: 1px solid #d0d0d0;
            border-radius: 6px;
            background-color: #f0f0f0;
            margin-top: 15px;
            padding-top: 10px;
            padding-bottom: 6px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #0078d7;
        }

        /* Input field styling */
        QLineEdit {
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            padding: 8px;
            background-color: #f0f0f0;
            selection-background-color: #0078d7;
            min-height: 16px;
        }

        QLineEdit:focus {
            border: 1px solid #0078d7;
            background-color: #f0f7ff;
        }

        QLineEdit:disabled {
            background-color: #f0f0f0;
            color: #888888;
        }

        QLineEdit::placeholder {
            color: #aaaaaa;
        }

        /* Standard buttons */
        QPushButton {
            background-color: #0078d7;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
            min-height: 16px;
        }

        QPushButton:hover {
            background-color: #106ebe;
        }

        QPushButton:pressed {
            background-color: #005a9e;
        }

        QPushButton:focus {
            outline: none;
        }

        QPushButton:disabled {
            background-color: #cccccc;
            color: #888888;
        }

        QPushButton:checked {
            background-color: #005a9e;
            border: 1px solid #0078d7;
        }

        /* Browse button */
        QPushButton#browse_button {
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #d0d0d0;
            font-weight: normal;
        }

        QPushButton#browse_button:hover {
            background-color: #e0e0e0;
        }

        QPushButton#browse_button:pressed {
            background-color: #d0d0d0;
        }

        /* Generate config button */
        QPushButton#generate_config {
            background-color: #107c10;
            font-weight: bold;
            padding: 10px 20px;
        }

        QPushButton#generate_config:hover {
            background-color: #0b6a0b;
        }

        QPushButton#generate_config:pressed {
            background-color: #004b00;
        }

        /* Refresh button */
        QPushButton#refresh_button {
            background-color: #0078d7;
            padding: 8px 16px;
        }

        /* Toggle buttons / orientation buttons */
        QPushButton#orientation_button, 
        QPushButton[checkable="true"] {
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #d0d0d0;
            padding: 6px 12px;
            font-weight: normal;
        }

        QPushButton#orientation_button:hover, 
        QPushButton[checkable="true"]:hover {
            background-color: #f0f0f0;
        }

        QPushButton#orientation_button:checked,
        QPushButton[checkable="true"]:checked {
            background-color: #0078d7;
            color: white;
            border: 1px solid #005a9e;
        }

        /* BPM/Optical flow toggle buttons */
        QPushButton#mode_button {
            padding: 10px 20px;
            font-size: 11pt;
        }

        QPushButton#mode_button:checked {
            background-color: #005a9e;
            border-bottom: 3px solid white;
        }

        /* Separator line */
        QFrame[frameShape="4"], 
        QFrame[frameShape="5"], 
        QFrame.separator_line {
            background-color: #f0f0f0;
            max-height: 1px;
            min-height: 1px;
            border: none;
        }

        /* Form layout spacing */
        QFormLayout {
            spacing: 12px;
        }

        /* Form field labels */
        QLabel.field_label, 
        QLabel {
            font-weight: 500;
            background-color: #f0f0f0
        }

        /* Preview panel styling */
        QLabel#image_preview {
            background-color: white;
            border: 1px solid #f0f0f0;
            border-radius: 4px;
        }

        /* Tooltips */
        QToolTip {
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #ddddcc;
            padding: 3px;
        }
        QCheckBox {
            spacing: 8px;
            background-color: #f0f0f0;
            padding: 4px;
        }

        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }

        QCheckBox::indicator:unchecked {
            border: 1px solid #d0d0d0;
            background-color: white;
            border-radius: 3px;
        }

        QCheckBox::indicator:checked {
            border: 1px solid #0078d7;
            background-color: #0078d7;
            border-radius: 3px;
            image: url(assets/checkmark.png); /* You'll need to provide this image or use a different approach */
        }

        QCheckBox:hover {
            background-color: #e8e8e8;
        }
        """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_panel_max_height()

    def update_panel_max_height(self):
        self.opticalflow_mode_config_panel.setMaximumHeight(int(self.height() * 0.6))
        self.bpm_mode_config_panel.setMaximumHeight(int(self.height() * 0.85))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def convert_cvimg_to_qimg(cvImg):
    height, width, _ = cvImg.shape
    bytesPerLine = 3 * width
    data = cvImg.tobytes()
    qImg = QImage(data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
    return qImg


if __name__ == "__main__":
    key_map = {
        "Feed path": "target",
        "Feed FPS": "fps",
        "Real RPM": "real_rpm",
        "Quadrant": "quadrant",
        "Number of boxes": "target_num_boxes",
        "Box size": "target_box_size",
        "Start from box": "start_from_box",
        "Trim last N boxes": "trim_last_n_boxes",
        "Frame buffer size": "frame_buffer_size",
        "Update frequency": "color_delta_update_frequency",
        "Contrast multiplier override": "contrast_multiplier",
        "Detection threshold multiplier": "threshold_multiplier",
        "Detection warmup frames": "detection_warmup_frames",
        "Kernel size": "erosion_dilation_kernel_size",
        "Dilation iterations": "dilation_iterations",
        "Erosion iterations": "erosion_iterations",
        "Run ID": "id",
        "Ground angle": "ground_angle",
        "Pixel tracking threshold": "pixel_threshold",
        "Deadzone offset x": "deadzone_offset_x",
        "Deadzone offset y": "deadzone_offset_y",
        "Deadzone shape": "deadzone_shape",
        "RPM smoothing": "rpm_buffer_length",
        "RPM acceleration bound": "rpm_acceleration_bound",
        "RPM zero timeout seconds": "rpm_zero_timeout_seconds",
        "Turbine diameter": "turbine_diameter",
    }
    main()
