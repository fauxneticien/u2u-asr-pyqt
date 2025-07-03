import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QProgressBar,
    QFileDialog, QLabel, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread
import os
from time import sleep

# Worker class to process the file in a separate thread
class WordCountWorker(QObject):
    progress = pyqtSignal(int)          # Signal to update progress bar
    finished = pyqtSignal(list)         # Signal to emit result when done

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath        # Path of file to be processed

    def run(self):
        results = []                    # Store word counts for each line
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                total_lines = len(lines)

                for i, line in enumerate(lines):
                    word_count = len(line.strip().split())  # Count words in line
                    results.append(f"Line {i + 1}: {word_count} words")
                    progress_percent = int((i + 1) / total_lines * 100)
                    self.progress.emit(progress_percent)     # Update progress bar

                    # Add delay to see progress bar
                    sleep(1)

        except Exception as e:
            results = [f"Error reading file: {str(e)}"]
        self.finished.emit(results)     # Emit final results when done


# Main application window
class WordCountApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Word Count per Line with Progress")
        self.setGeometry(200, 200, 500, 150)

        # UI Components
        layout = QVBoxLayout()
        self.label = QLabel("Select a text file to begin")
        self.progress_bar = QProgressBar()
        self.select_button = QPushButton("Select Text File")

        # Add widgets to layout
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.select_button)
        self.setLayout(layout)

        # Connect button to method
        self.select_button.clicked.connect(self.select_file)

    def select_file(self):
        # Open file dialog to choose a text file
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt)")
        if filepath:
            self.label.setText(f"Processing file: {os.path.basename(filepath)}")
            self.run_worker(filepath)

    def run_worker(self, filepath):
        # Disable button while processing
        self.select_button.setEnabled(False)
        self.progress_bar.setValue(0)

        # Set up worker and thread
        self.thread = QThread()
        self.worker = WordCountWorker(filepath)
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
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Output File", "", "Text Files (*.txt)")
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(results))
                QMessageBox.information(self, "Success", "Output saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {e}")
        else:
            QMessageBox.warning(self, "Canceled", "Save operation was canceled.")

        self.label.setText("Select a text file to begin")
        self.progress_bar.setValue(0)


# Main execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WordCountApp()
    window.show()
    sys.exit(app.exec_())
