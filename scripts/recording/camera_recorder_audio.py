#!/usr/bin/env python3
"""
camera_recorder_audio.py

Enhanced camera recorder with audio capture from virtual audio device.
Captures both video and audio (e.g., from TTS) into a single video file.
Now uses BaseVideoRecorder.
"""

import cv2
import numpy as np
import argparse
import time
import sounddevice as sd
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from scripts.recording.video_recorder_base import BaseVideoRecorder


class CameraRecorderWithAudio(BaseVideoRecorder):
    """Records USB camera with Redis annotations and captures audio."""

    def __init__(
        self,
        camera_id: int = 0,
        stream_name: str = "annotated_camera",
        host: str = "localhost",
        port: int = 6379,
        output_file: Optional[str] = None,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        codec: str = "mp4v",
        layout: str = "side-by-side",
        audio_device: Optional[str] = None,
        audio_samplerate: int = 44100,
    ):
        # We need two files for this one initially
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.final_output = f"recording_{timestamp}.mp4"
            self.video_file = f"video_{timestamp}.mp4"
            self.audio_file = f"audio_{timestamp}.wav"
        else:
            self.final_output = output_file
            base_name = Path(output_file).stem
            self.video_file = f"{base_name}_video.mp4"
            self.audio_file = f"{base_name}_audio.wav"

        super().__init__(
            camera_id=camera_id,
            stream_name=stream_name,
            host=host,
            port=port,
            fps=fps,
            width=width,
            height=height,
            output_file=self.video_file,
            codec=codec,
        )

        self.layout = layout
        self.audio_samplerate = audio_samplerate

        # Initialize video writer (without audio initially)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_width = width * 2 if layout == "side-by-side" else width
        output_height = height

        self.video_writer = cv2.VideoWriter(self.video_file, fourcc, fps, (output_width, output_height))
        if not self.video_writer.isOpened():
            self.camera.release()
            raise RuntimeError("Failed to create video writer")

        # Initialize audio capture
        self.audio_queue = queue.Queue()
        self.audio_device = self._find_audio_device(audio_device)
        self.audio_frames = []

        cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Recording", output_width, output_height)

    def _find_audio_device(self, device_name: Optional[str] = None):
        """Find audio input device by name or use default."""
        if device_name is None:
            return sd.default.device[0]

        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                if device_name.lower() in device["name"].lower():
                    return i

        return sd.default.device[0]

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback for sounddevice stream."""
        if status:
            print(f"Audio status: {status}")

        if not self.paused:
            self.audio_queue.put(indata.copy())

    def _audio_recording_thread(self):
        """Thread that handles audio recording."""
        try:
            with sd.InputStream(
                device=self.audio_device,
                channels=1,
                samplerate=self.audio_samplerate,
                callback=self._audio_callback,
                blocksize=int(self.audio_samplerate / self.fps),
            ):
                while self.recording:
                    try:
                        audio_data = self.audio_queue.get(timeout=1.0)
                        self.audio_frames.append(audio_data)
                    except queue.Empty:
                        continue

        except Exception as e:
            print(f"✗ Audio recording error: {e}")

    def draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw recording info overlay."""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        cv2.rectangle(overlay, (0, 0), (width, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)

        y_pos = 20
        status = "PAUSED" if self.paused else "REC"
        status_color = (0, 165, 255) if self.paused else (0, 0, 255)
        cv2.circle(frame, (15, y_pos - 5), 6, status_color, -1)
        cv2.putText(frame, status, (30, y_pos), font, 0.5, color, 1)

        elapsed = time.time() - self.start_time
        minutes, seconds = int(elapsed // 60), int(elapsed % 60)
        text = f"Frames: {self.frame_count} | Time: {minutes:02d}:{seconds:02d}"
        cv2.putText(frame, text, (110, y_pos), font, 0.5, color, 1)

        y_pos += 22
        text = f"Camera: {self.camera_id} | Stream: {self.stream_name}"
        cv2.putText(frame, text, (10, y_pos), font, 0.5, color, 1)

        return frame

    def process_frame(self, camera_frame: np.ndarray, annotated_frame: Optional[np.ndarray]) -> np.ndarray:
        """Combine camera and annotated frames."""
        camera_frame = self.resize_frame(camera_frame)

        if annotated_frame is None:
            if self.last_annotated_frame is not None:
                annotated_frame = self.last_annotated_frame
            else:
                annotated_frame = self.create_placeholder_frame()
        else:
            annotated_frame = self.resize_frame(annotated_frame)
            self.last_annotated_frame = annotated_frame.copy()

        if self.layout == "side-by-side":
            combined = np.hstack([camera_frame, annotated_frame])
        else:
            combined = camera_frame.copy()
            overlay = cv2.resize(annotated_frame, (self.width // 2, self.height // 2))
            y_offset = self.height - overlay.shape[0] - 10
            x_offset = self.width - overlay.shape[1] - 10
            roi = combined[y_offset : y_offset + overlay.shape[0], x_offset : x_offset + overlay.shape[1]]
            blended = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)
            combined[y_offset : y_offset + overlay.shape[0], x_offset : x_offset + overlay.shape[1]] = blended

        return self.draw_info_overlay(combined)

    def run(self):
        """Main recording loop."""
        audio_thread = threading.Thread(target=self._audio_recording_thread, daemon=True)
        audio_thread.start()

        try:
            while self.recording:
                ret, camera_frame = self.camera.read()
                if not ret:
                    break

                result = self.image_streamer.get_latest_image()
                annotated_frame = result[0] if result else None

                display_frame = self.process_frame(camera_frame, annotated_frame)

                if not self.paused:
                    # We write the raw combined frame without info overlay to the file
                    # Wait, the original script wrote combined_frame, which didn't have info overlay
                    # Actually, draw_info_overlay modifies the frame in place.
                    # Let's re-read the original.
                    # The original:
                    # combined_frame = self.combine_frames(camera_frame, annotated_frame)
                    # display_frame = self.draw_info_overlay(combined_frame)
                    # if not self.paused:
                    #     self.video_writer.write(combined_frame)
                    # Since draw_info_overlay modified combined_frame, it wrote it with overlay.

                    self.video_writer.write(display_frame)
                    self.frame_count += 1

                cv2.imshow("Recording", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    self.recording = False
                elif key == ord("p"):
                    self.paused = not self.paused

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup and merge audio/video."""
        self.video_writer.release()

        # Save audio to WAV
        if self.audio_frames:
            import scipy.io.wavfile as wavfile
            audio_data = np.concatenate(self.audio_frames, axis=0)
            wavfile.write(self.audio_file, self.audio_samplerate, audio_data)

            # Merge video and audio using ffmpeg
            import subprocess
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", self.video_file, "-i", self.audio_file,
                        "-c:v", "copy", "-c:a", "aac", "-strict", "experimental",
                        self.final_output,
                    ],
                    check=True, capture_output=True,
                )
                Path(self.video_file).unlink()
                Path(self.audio_file).unlink()
            except Exception as e:
                print(f"✗ Failed to merge: {e}")

        super().cleanup()


def main():
    parser = argparse.ArgumentParser(description="Record camera with audio")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--stream", type=str, default="annotated_camera")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--layout", type=str, default="side-by-side", choices=["side-by-side", "overlay"])
    parser.add_argument("--audio-device", type=str, default=None)
    args = parser.parse_args()

    recorder = CameraRecorderWithAudio(
        camera_id=args.camera,
        stream_name=args.stream,
        host=args.host,
        port=args.port,
        output_file=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        layout=args.layout,
        audio_device=args.audio_device,
    )
    recorder.run()

if __name__ == "__main__":
    main()
