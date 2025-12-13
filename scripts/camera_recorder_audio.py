#!/usr/bin/env python3
"""
Enhanced camera recorder with audio capture from virtual audio device.
Captures both video and audio (e.g., from TTS) into a single video file.
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
from redis_robot_comm import RedisImageStreamer


class CameraRecorderWithAudio:
    """Records USB camera with Redis annotations and captures audio."""

    def __init__(
        self,
        camera_id: int = 0,
        stream_name: str = "annotated_camera",
        host: str = "localhost",
        port: int = 6379,
        output_file: str = None,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        codec: str = "mp4v",
        layout: str = "side-by-side",
        audio_device: str = None,
        audio_samplerate: int = 44100,
    ):
        """
        Initialize recorder with audio capture.

        Args:
            audio_device: Name or index of audio input device (e.g., "Virtual_Speaker.monitor")
            audio_samplerate: Audio sample rate in Hz
        """
        self.camera_id = camera_id
        self.stream_name = stream_name
        self.fps = fps
        self.width = width
        self.height = height
        self.layout = layout
        self.audio_samplerate = audio_samplerate

        # Generate output filename
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_file = f"video_{timestamp}.mp4"
            self.audio_file = f"audio_{timestamp}.wav"
            self.output_file = f"recording_{timestamp}.mp4"
        else:
            base_name = Path(output_file).stem
            self.video_file = f"{base_name}_video.mp4"
            self.audio_file = f"{base_name}_audio.wav"
            self.output_file = output_file

        # Initialize camera
        print(f"Opening camera {camera_id}...")
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        print("✓ Camera opened")

        # Initialize Redis
        try:
            self.streamer = RedisImageStreamer(host=host, port=port, stream_name=stream_name)
            print(f"✓ Connected to Redis: {stream_name}")
        except Exception as e:
            self.camera.release()
            raise RuntimeError(f"Failed to connect to Redis: {e}")

        # Initialize video writer (without audio initially)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_width = width * 2 if layout == "side-by-side" else width
        output_height = height

        self.video_writer = cv2.VideoWriter(self.video_file, fourcc, fps, (output_width, output_height))
        if not self.video_writer.isOpened():
            self.camera.release()
            raise RuntimeError("Failed to create video writer")

        print(f"✓ Video writer initialized: {self.video_file}")

        # Initialize audio capture
        self.audio_queue = queue.Queue()
        self.audio_device = self._find_audio_device(audio_device)
        self.audio_frames = []

        print(f"✓ Audio device: {self.audio_device}")
        print(f"  Sample rate: {audio_samplerate} Hz")

        # State
        self.recording = True
        self.paused = False
        self.frame_count = 0
        self.start_time = time.time()
        self.last_annotated_frame = None

        cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Recording", output_width, output_height)

    def _find_audio_device(self, device_name: str = None):
        """Find audio input device by name or use default."""
        if device_name is None:
            return sd.default.device[0]  # Default input

        # Search by name
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                if device_name.lower() in device["name"].lower():
                    print(f"  Found audio device: {device['name']}")
                    return i

        print(f"⚠ Device '{device_name}' not found, using default")
        return sd.default.device[0]

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback for sounddevice stream."""
        if status:
            print(f"Audio status: {status}")

        if not self.paused:
            # Copy audio data to queue
            self.audio_queue.put(indata.copy())

    def _audio_recording_thread(self):
        """Thread that handles audio recording."""
        try:
            with sd.InputStream(
                device=self.audio_device,
                channels=1,
                samplerate=self.audio_samplerate,
                callback=self._audio_callback,
                blocksize=int(self.audio_samplerate / self.fps),  # Match video frame rate
            ):
                print("✓ Audio recording started")

                while self.recording:
                    try:
                        # Get audio data from queue
                        audio_data = self.audio_queue.get(timeout=1.0)
                        self.audio_frames.append(audio_data)
                    except queue.Empty:
                        continue

        except Exception as e:
            print(f"✗ Audio recording error: {e}")

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target dimensions."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            return cv2.resize(frame, (self.width, self.height))
        return frame

    def create_placeholder_frame(self, text: str = "Waiting for annotated frames...") -> np.ndarray:
        """Create placeholder when annotated frame unavailable."""
        placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, 0.7, 2)
        x = (self.width - text_width) // 2
        y = (self.height + text_height) // 2
        cv2.putText(placeholder, text, (x, y), font, 0.7, (128, 128, 128), 2)
        return placeholder

    def combine_frames(self, camera_frame: np.ndarray, annotated_frame: np.ndarray = None) -> np.ndarray:
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
            return np.hstack([camera_frame, annotated_frame])
        else:  # overlay
            combined = camera_frame.copy()
            overlay = cv2.resize(annotated_frame, (self.width // 2, self.height // 2))
            y_offset = self.height - overlay.shape[0] - 10
            x_offset = self.width - overlay.shape[1] - 10
            roi = combined[y_offset : y_offset + overlay.shape[0], x_offset : x_offset + overlay.shape[1]]
            blended = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)
            combined[y_offset : y_offset + overlay.shape[0], x_offset : x_offset + overlay.shape[1]] = blended
            return combined

    def draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw recording info overlay."""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        cv2.rectangle(overlay, (0, 0), (width, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)

        # Status
        y_pos = 20
        status = "PAUSED" if self.paused else "REC"
        status_color = (0, 165, 255) if self.paused else (0, 0, 255)
        cv2.circle(frame, (15, y_pos - 5), 6, status_color, -1)
        cv2.putText(frame, status, (30, y_pos), font, 0.5, color, 1)

        # Time and frames
        elapsed = time.time() - self.start_time
        minutes, seconds = int(elapsed // 60), int(elapsed % 60)
        text = f"Frames: {self.frame_count} | Time: {minutes:02d}:{seconds:02d}"
        cv2.putText(frame, text, (110, y_pos), font, 0.5, color, 1)

        # Camera info
        y_pos += 22
        text = f"Camera: {self.camera_id} | Stream: {self.stream_name}"
        cv2.putText(frame, text, (10, y_pos), font, 0.5, color, 1)

        # Audio info
        y_pos += 22
        audio_frames = len(self.audio_frames)
        text = f"Audio: {audio_frames} frames | {self.audio_samplerate} Hz"
        cv2.putText(frame, text, (10, y_pos), font, 0.5, color, 1)

        # Output
        y_pos += 22
        text = f"Output: {self.output_file}"
        cv2.putText(frame, text, (10, y_pos), font, 0.5, color, 1)

        # Controls
        y_pos += 22
        text = "Q=Quit | P=Pause | S=Screenshot"
        cv2.putText(frame, text, (10, y_pos), font, 0.4, (200, 200, 200), 1)

        return frame

    def run(self):
        """Main recording loop."""
        print("\n" + "=" * 60)
        print("Recording started (with audio)")
        print("=" * 60 + "\n")

        # Start audio recording thread
        audio_thread = threading.Thread(target=self._audio_recording_thread, daemon=True)
        audio_thread.start()

        try:
            while self.recording:
                ret, camera_frame = self.camera.read()
                if not ret:
                    print("✗ Failed to read from camera")
                    break

                # Get annotated frame
                annotated_frame = None
                result = self.streamer.get_latest_image()
                if result:
                    annotated_frame, _ = result

                # Combine and display
                combined_frame = self.combine_frames(camera_frame, annotated_frame)
                display_frame = self.draw_info_overlay(combined_frame)

                if not self.paused:
                    self.video_writer.write(combined_frame)
                    self.frame_count += 1

                cv2.imshow("Recording", display_frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("\nStopping...")
                    self.recording = False
                elif key == ord("p"):
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Resumed'}")

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup and merge audio/video."""
        print("\nCleaning up...")

        self.camera.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

        # Save audio to WAV
        if self.audio_frames:
            print("Saving audio...")
            import scipy.io.wavfile as wavfile

            audio_data = np.concatenate(self.audio_frames, axis=0)
            wavfile.write(self.audio_file, self.audio_samplerate, audio_data)
            print(f"✓ Audio saved: {self.audio_file}")

            # Merge video and audio using ffmpeg
            print("Merging video and audio...")
            import subprocess  # nosec B404

            try:
                subprocess.run(  # nosec B603 B607
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        self.video_file,
                        "-i",
                        self.audio_file,
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-strict",
                        "experimental",
                        self.output_file,
                    ],
                    check=True,
                    capture_output=True,
                )

                print(f"✓ Final video with audio: {self.output_file}")

                # Cleanup temporary files
                Path(self.video_file).unlink()
                Path(self.audio_file).unlink()

            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to merge: {e}")
                print(f"Video: {self.video_file}")
                print(f"Audio: {self.audio_file}")

        # Print summary
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("RECORDING SUMMARY")
        print("=" * 60)
        print(f"  Frames:     {self.frame_count}")
        print(f"  Duration:   {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
        print(f"  Audio data: {len(self.audio_frames)} frames")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Record camera with audio capture")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--stream", type=str, default="annotated_camera")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--layout", type=str, default="side-by-side", choices=["side-by-side", "overlay"])
    parser.add_argument(
        "--audio-device", type=str, default=None, help="Audio input device name (e.g., 'Virtual_Speaker.monitor')"
    )
    parser.add_argument("--audio-rate", type=int, default=44100)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  CAMERA + AUDIO RECORDER")
    print("=" * 60)

    try:
        recorder = CameraRecorderWithAudio(
            camera_id=args.camera,
            stream_name=args.stream,
            host=args.host,
            port=args.port,
            output_file=args.output,
            fps=args.fps,
            width=args.width,
            height=args.height,
            codec=args.codec,
            layout=args.layout,
            audio_device=args.audio_device,
            audio_samplerate=args.audio_rate,
        )
        recorder.run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
