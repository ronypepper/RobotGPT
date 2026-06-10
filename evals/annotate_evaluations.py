"""
Copyright (c) 2026 ronypepper.
License: BSD-3-Clause
"""

import os
import sys
import yaml
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QSlider, QVBoxLayout, QHBoxLayout, QFileDialog
)
from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


class RobotGPTEvaluationAnnotator(QWidget):

    demo_paths: list[str]
    demo_annotations: dict
    current_demo: str

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RobotGPT Evaluation Annotator (SPACE: Pause/Play | A: 5x Speed | D: 10x Speed)")

        # Video player components
        self.video_widget = QVideoWidget()
        self.audio = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)

        # Mark demo success/failure/invalid buttons
        self.success_button = QPushButton("SUCCESS")
        self.success_button.clicked.connect(self.mark_demo_successful)
        self.failure_button = QPushButton("FAILURE")
        self.failure_button.clicked.connect(self.mark_demo_failed)
        self.invalid_button = QPushButton("INVALID")
        self.invalid_button.clicked.connect(self.mark_demo_invalid)

        # Video playback slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_playback_position)
        self.player.positionChanged.connect(self.update_slider_position)
        self.player.durationChanged.connect(self.update_slider_range)

        # Widget layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.invalid_button)
        buttons_layout.addWidget(self.failure_button)
        buttons_layout.addWidget(self.success_button)
        layout = QVBoxLayout()
        layout.addWidget(self.video_widget)
        layout.addWidget(self.slider)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

        # Let user select directory to evaluate after event loop starts
        QTimer.singleShot(0, self.select_directory)

    def select_directory(self):
        self.demo_annotations = {
            "demonstrations": []
        }

        self.dir_path = QFileDialog.getExistingDirectory(self, "Select dataset directory", "robotgpt_output/evaluation")
        if self.dir_path:
            self.demo_paths = []
            dir_entries = os.listdir(self.dir_path)
            for entry in dir_entries:
                if entry.endswith(".mp4") and entry.startswith("scene-episode-"):
                    self.demo_paths.append(entry)
            if len(self.demo_paths) == 0:
                self.close()
            else:
                self.open_next_video()
        else:
            self.close()

    def open_next_video(self):
        if len(self.demo_paths) == 0:
            self.process_and_save_annotations()
            self.select_directory()
        else:
            self.current_demo = self.demo_paths.pop()
            video_path = os.path.join(self.dir_path, self.current_demo)
            if not os.path.exists(video_path):
                self.open_next_video()
            else:
                self.player.setSource(QUrl.fromLocalFile(video_path))
                self.player.play()

    def set_playback_position(self, position):
        self.player.setPosition(position)

    def update_slider_position(self, position):
        self.slider.setValue(position)

    def update_slider_range(self, range):
        self.slider.setRange(0, range)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_playback()
            event.accept()
        elif event.key() == Qt.Key.Key_A:
            self.player.setPlaybackRate(5.0)
        elif event.key() == Qt.Key.Key_D:
            self.player.setPlaybackRate(10.0)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_A or event.key() == Qt.Key.Key_D:
            self.player.setPlaybackRate(1.0)
        else:
            super().keyReleaseEvent(event)

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def mark_demo_successful(self):
        self.annotate_demo(invalid=False, success=True)
        self.open_next_video()

    def mark_demo_failed(self):
        self.annotate_demo(invalid=False, success=False)
        self.open_next_video()

    def mark_demo_invalid(self):
        self.annotate_demo(invalid=True)
        self.open_next_video()

    def annotate_demo(self, invalid: bool, success: bool = False):
        result = "INVALID" if invalid else "SUCCESS" if success else "FAILURE"
        self.demo_annotations["demonstrations"].append({
                "name": self.current_demo,
                "result": result,
                "duration_s": self.player.position() / 1000.0 if success else 0.0
            })

    def process_and_save_annotations(self):
        # Process annotations statistics
        num_success, num_failure, num_invalid = 0, 0, 0
        avg_duration = 0.0
        for demo in self.demo_annotations["demonstrations"]:
            match demo["result"]:
                case "SUCCESS":
                    num_success += 1
                    avg_duration += demo["duration_s"]
                case "FAILURE":
                    num_failure += 1
                case "INVALID":
                    num_invalid += 1

        if num_success > 0:
            avg_duration /= num_success

        success_percentage = 0.0
        if num_success + num_failure > 0:
            success_percentage = num_success / (num_success + num_failure) * 100.0

        # Add annotations statistics
        self.demo_annotations["num_demos_total"] = num_success + num_failure + num_invalid
        self.demo_annotations["num_success"] = num_success
        self.demo_annotations["num_failure"] = num_failure
        self.demo_annotations["num_invalid"] = num_invalid
        self.demo_annotations["success_percentage"] = success_percentage
        self.demo_annotations["average_duration_s"] = avg_duration

        # Save to yaml
        with open(os.path.join(self.dir_path, "annotations.yaml"), "w") as f:
            yaml.dump(self.demo_annotations, f, sort_keys=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotGPTEvaluationAnnotator()
    window.resize(1200, 700)
    window.show()
    sys.exit(app.exec())
