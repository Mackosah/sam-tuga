# sam-tuga
import numpy as np
import time
import torch
import torch.nn as nn
from collections import deque
from scipy.stats import mode

# --- CONFIG ---
WINDOW_SIZE = 16
STRIDE = 8
SMOOTHING_WINDOW = 3
SIMULATED_TOTAL_FRAMES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ACTUAL MODEL PLACEHOLDER (REPLACE THIS WITH YOUR MODEL) ---
class My3DActionModel(nn.Module):
    def __init__(self, input_shape=(16, 1024, 3), num_classes=5):
        super(My3DActionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# --- PREPROCESSING FUNCTION ---
def preprocess_patch(patch):
    """
    Convert patch to a PyTorch tensor suitable for the model.
    Assumes patch is list of np arrays: [(1024, 3), ...] of length 16
    Output shape: (1, 16, 1024, 3)
    """
    tensor = torch.from_numpy(np.stack(patch)).unsqueeze(0).float()  # shape (1, 16, 1024, 3)
    return tensor.to(DEVICE)

# --- PREDICTION DECODING ---
def decode_prediction(logits):
    return logits.argmax(dim=1).item()

# --- REAL-TIME INFERENCE STREAM ---
def realtime_inference_stream(frame_generator, model, window_size, stride, smoothing_window):
    buffer = deque(maxlen=window_size)
    prediction_history = deque(maxlen=smoothing_window)

    model.eval()

    for frame in frame_generator:
        buffer.append(frame)

        if len(buffer) == window_size:
            t_patch = list(buffer)
            input_tensor = preprocess_patch(t_patch)

            with torch.no_grad():
                logits = model(input_tensor)
            pred = decode_prediction(logits)

            prediction_history.append(pred)
            smoothed_pred = mode(prediction_history)[0][0]

            print(f"Raw: {pred}, Smoothed: {smoothed_pred}")
            time.sleep(0.05)  # simulate real-time inference delay

# --- SIMULATED FRAME GENERATOR ---
def simulated_frame_generator(total_frames=SIMULATED_TOTAL_FRAMES):
    for _ in range(total_frames):
        yield np.random.rand(1024, 3)

# --- MAIN ---
if __name__ == "__main__":
    model = My3DActionModel().to(DEVICE)
    frame_stream = simulated_frame_generator()
    realtime_inference_stream(
        frame_generator=frame_stream,
        model=model,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        smoothing_window=SMOOTHING_WINDOW
    )
