import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QProgressBar,
    QFileDialog, QLabel, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread
import os

import numpy as np
from pedalboard.io import AudioFile
from silero_vad import SileroVAD
from mhubert_asr import mHuBERTASR
from pympi import Elan

basedir = os.path.dirname(__file__)

# Worker class to process the file in a separate thread
class WordCountWorker(QObject):
    progress = pyqtSignal(int)          # Signal to update progress bar
    finished = pyqtSignal(list)         # Signal to emit result when done

    def __init__(self, filepath, vad_model, asr_model):
        super().__init__()
        self.filepath = filepath        # Path of file to be processed

        self.vad_model = vad_model
        self.asr_model = asr_model

    def run(self):
        try:
            with AudioFile(self.filepath).resampled_to(16_000) as f:
                wav = f.read(f.frames)

                if wav.shape[0] != 1:
                    wav = np.mean(wav, axis=0, keepdims=True)

            speech_timestamps = self.vad_model.get_speech_timestamps(
                wav,
                progress_tracking_callback=lambda x: self.progress.emit(x)
            )

            speech_timestamps_with_text = self.asr_model.predict_text(
                wav,
                speech_timestamps,
                progress_tracking_callback=lambda x: self.progress.emit(x)
            )

            eaf_data = Elan.Eaf()
            eaf_data.add_linked_file(self.filepath)
            # Remove 'default' tier from newly created eaf object
            eaf_data.remove_tier('default')
            eaf_data.add_tier(f"Channel 0")

            for a in speech_timestamps_with_text:
                start_ms = int(a["start"] / 16)
                end_ms   = int(a["end"] / 16)
                eaf_data.add_annotation(f"Channel 0", start=start_ms, end=end_ms, value=a["text"])

        except Exception as e:
            results = [f"Error reading file: {str(e)}"]
        self.finished.emit([ eaf_data ])     # Emit final results when done

# Main application window
class WordCountApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mixtec wav-to-eaf")
        self.setGeometry(200, 200, 500, 150)

        # UI Components
        layout = QVBoxLayout()
        self.label = QLabel("Select a wav file to begin")
        self.progress_bar = QProgressBar()
        self.select_button = QPushButton("Select .wav File")

        self.vad_model = SileroVAD(basedir + "/assets/silero_vad.onnx")
        self.asr_model = mHuBERTASR(basedir + "/assets/mhubert_asr.onnx")

        # Add widgets to layout
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.select_button)
        self.setLayout(layout)

        # Connect button to method
        self.select_button.clicked.connect(self.select_file)

    def select_file(self):
        # Open file dialog to choose a text file
        filepath, _ = QFileDialog.getOpenFileName(self, "Open wav File", "", "Wav Files (*.wav)")
        if filepath:
            self.label.setText(f"Processing file: {os.path.basename(filepath)}")
            self.run_worker(filepath)

    def run_worker(self, filepath):
        # Disable button while processing
        self.select_button.setEnabled(False)
        self.progress_bar.setValue(0)

        # Set up worker and thread
        self.thread = QThread()
        self.worker = WordCountWorker(filepath, self.vad_model, self.asr_model)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(lambda: self.select_button.setEnabled(True))

        self.thread.start()

    def on_finished(self, results):
        # Let user choose where to save the output
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Output File", "", "ELAN Files (*.eaf)")
        if save_path:
            try:
                results[0].to_file(save_path)
                QMessageBox.information(self, "Success", "Output saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {e}")
        else:
            QMessageBox.warning(self, "Canceled", "Save operation was canceled.")

        self.label.setText("Select a wav file to begin")
        self.progress_bar.setValue(0)


# Main execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WordCountApp()
    window.show()
    sys.exit(app.exec_())
