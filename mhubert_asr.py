import json

import numpy as np
import onnxruntime as ort

from itertools import groupby

class mHuBERTASR:

    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)

        meta = self.session.get_modelmeta()
        self.vocab = list(json.loads(meta.custom_metadata_map['vocab']).values())

    def predict_text(self, wav, speech_timestamps, progress_tracking_callback=None):

        for i, timestamp in enumerate(speech_timestamps):

            wav_chunk = wav[:, timestamp['start']:timestamp['end']]

            inputs = { self.session.get_inputs()[0].name: wav_chunk }
            outputs = self.session.run(None, inputs)

            indices = [ int(i) for i in np.argmax(outputs[0][0], axis=-1) ]
            unique_indices = [key for key, _ in groupby(indices) ]
            text = "".join([ char for index in unique_indices if (char := self.vocab[index]) != '<pad>' ])

            speech_timestamps[i]["text"] = text

            if progress_tracking_callback:
                progress_percent_int = 50 + int((i + 1)/len(speech_timestamps)*50)
                progress_tracking_callback(progress_percent_int)

        return speech_timestamps
