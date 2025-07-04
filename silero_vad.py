import numpy as np
import onnxruntime

class SileroVAD:
    """
    Modified Silero-VAD OnnxWrapper without torch/torchaudio dependencies
    Adapted from: https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
    """

    def __init__(self, onnx_path):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'], sess_options=opts)
        
        # Hard-code values for 16 kHz model (don't need to support both 8 and 16 kHz models)
        self.sampling_rate = 16_000
        self.batch_size = 1
        self.context_size = 64
        self.window_size_samples = 512

        self.reset_states()

    def reset_states(self):
        self._state = np.zeros((2, self.batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0)

    def model(self, x):

        if not len(self._context):
            self._context = np.zeros((self.batch_size, self.context_size))

        _input = np.zeros((self.batch_size, self.window_size_samples + self.context_size), dtype=np.float32)
        _input[:, :self.context_size] = self._context
        _input[:, self.context_size:] = x

        ort_inputs = { 'input': _input, 'state': self._state, 'sr': np.array(16_000, dtype='int64') }
        ort_outs = self.session.run(None, ort_inputs)
        out, state = ort_outs
        
        self._state = state
        self._context = x[..., -self.context_size:]

        return out

    def get_speech_timestamps(
        self,
        wav,
        # Keep all variable names and defaults from Silero's get_speech_timestamps()
        # https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py#L191
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float('inf'),
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        return_seconds: bool = False,
        time_resolution: int = 1,
        progress_tracking_callback=None,
        neg_threshold: float = None
    ):
        audio_length_samples = wav.shape[1]

        speech_probs = []

        chunk_starts = list(range(0, wav.shape[1], self.window_size_samples))

        for i, chunk_start in enumerate(chunk_starts):
            chunk = wav[:, chunk_start:chunk_start+self.window_size_samples]

            if chunk.shape[1] < self.window_size_samples:
                break

            speech_prob = self.model(chunk).item()
            speech_probs.append(speech_prob)

            if progress_tracking_callback:
                progress_percent_int = int((i + 1)/len(chunk_starts)*50)
                progress_tracking_callback(progress_percent_int)

        # reset model states after each audio
        self.reset_states()

        sampling_rate = self.sampling_rate
        min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
        speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        max_speech_samples = sampling_rate * max_speech_duration_s - self.window_size_samples - 2 * speech_pad_samples
        min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        min_silence_samples_at_max_speech = sampling_rate * 98 / 1000        

        # All detection logic below copy and pasted from Silero
        # https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py#L310

        triggered = False
        speeches = []
        current_speech = {}

        if neg_threshold is None:
            neg_threshold = max(threshold - 0.15, 0.01)

        temp_end = 0  # to save potential segment end (and tolerate some silence)
        prev_end = next_start = 0  # to save potential segment limits in case of maximum segment size reached

        for i, speech_prob in enumerate(speech_probs):
            if (speech_prob >= threshold) and temp_end:
                temp_end = 0
                if next_start < prev_end:
                    next_start = self.window_size_samples * i

            if (speech_prob >= threshold) and not triggered:
                triggered = True
                current_speech['start'] = self.window_size_samples * i
                continue

            if triggered and (self.window_size_samples * i) - current_speech['start'] > max_speech_samples:
                if prev_end:
                    current_speech['end'] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:  # previously reached silence (< neg_thres) and is still not speech (< thres)
                        triggered = False
                    else:
                        current_speech['start'] = next_start
                    prev_end = next_start = temp_end = 0
                else:
                    current_speech['end'] = self.window_size_samples * i
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = self.window_size_samples * i
                if ((self.window_size_samples * i) - temp_end) > min_silence_samples_at_max_speech:  # condition to avoid cutting in very short silence
                    prev_end = temp_end
                if (self.window_size_samples * i) - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech['end'] = temp_end
                    if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                        speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

        if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
            current_speech['end'] = audio_length_samples
            speeches.append(current_speech)

        for i, speech in enumerate(speeches):
            if i == 0:
                speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i+1]['start'] - speech['end']
                if silence_duration < 2 * speech_pad_samples:
                    speech['end'] += int(silence_duration // 2)
                    speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
                else:
                    speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                    speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

        if return_seconds:
            audio_length_seconds = audio_length_samples / sampling_rate
            for speech_dict in speeches:
                speech_dict['start'] = max(round(speech_dict['start'] / sampling_rate, time_resolution), 0)
                speech_dict['end'] = min(round(speech_dict['end'] / sampling_rate, time_resolution), audio_length_seconds)

        return speeches
