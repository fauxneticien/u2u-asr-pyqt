# u2u-asr-pyqt

Experimental PyQt interface for packaging and deploying an ASR model on a user's computer using:

- [pedalboard](https://github.com/spotify/pedalboard) for dependency-free audio file opening and resampling (i.e. no need for ffmpeg, etc.)
- [silero](https://github.com/snakers4/silero-vad) for voice activity detection
- [mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147) fine-tuned for ASR (exported to ONNX, to remove torch/torchaudio dependency)
- PyQt5 for user interface PyInstaller for packaging models and interface into a single executable

![](README.gif)

## Setup

```bash
python -m venv packenv
source packenv/bin/activate
pip3 install PyQt5 PyInstaller pedalboard onnxruntime pympi-ling
rm -rf __pycache__

packenv/bin/pyinstaller \
	--windowed \
	--onedir \
	--add-data="assets/silero_vad.onnx:assets" \
	--add-data="assets/mhubert_asr.onnx:assets" \
	app.py
```