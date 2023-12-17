import time
import datetime
from pathlib import Path
import cv2

def frames_to_video(frames_path, output_video_path, fps=30):
    frames = [f for f in Path(frames_path).iterdir() if f.suffix == '.png']
    frames.sort(key=lambda x: int(x.stem.split('_')[1]))

    # Read the first frame to get dimensions
    first_frame = cv2.imread(str(frames[0]))
    height, width, layers = first_frame.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Write frames to the video
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

def get_current_date_time_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


class Timer():
    def __init__(self):
        self.times = []

    def start(self):
        self.t = time.time()

    def print(self, msg=''):
        print(f"Time taken: {msg}", time.time() - self.t)

    def get(self):
        return time.time() - self.t
    
    def store(self):
        self.times.append(time.time() - self.t)

    def average(self):
        return sum(self.times) / len(self.times)