#!/usr/bin/env python3
"""
record_camera_with_overlays.py

Enhanced recording script with text overlays showing:
- Current user task (persistent)
- Robot speech/explanations (timed, 4-5 seconds)
- TH Köln branding

Supports two layouts:
- Side-by-side: 1280x720 (camera + annotations + text below + logo)
- Overlay: 640x480 (annotations overlaid on camera + text overlay + logo)
"""

import cv2
import numpy as np
import argparse
import time
import threading
from datetime import datetime
from pathlib import Path
from collections import deque
from redis_robot_comm import RedisImageStreamer, RedisTextOverlayManager


class VideoOverlayRenderer:
    """Handles rendering of text overlays and branding on video frames."""

    def __init__(self, width: int, height: int, layout: str = "side-by-side"):
        """
        Initialize overlay renderer.

        Args:
            width: Video width
            height: Video height
            layout: 'side-by-side' or 'overlay'
        """
        self.width = width
        self.height = height
        self.layout = layout

        # Load logo (placeholder for now)
        self.logo = self._create_placeholder_logo()

        # Text settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_task = 0.6
        self.font_scale_speech = 0.55
        self.font_scale_branding = 0.4
        self.thickness = 2
        self.line_height = 25

        # Colors
        self.color_task = (255, 255, 255)  # White
        self.color_speech = (100, 255, 100)  # Light green
        self.color_branding = (200, 200, 200)  # Light gray

        # Speech history (for timed display)
        self.speech_queue = deque(maxlen=5)
        self.current_user_task = None

    def _create_placeholder_logo(self) -> np.ndarray:
        """Load TH Köln logo."""
        try:
            logo = cv2.imread("thkoelnlogo.png")
            if logo is None:
                return self._create_fallback_logo()
            return cv2.resize(logo, (150, 50))
        except Exception:
            return self._create_fallback_logo()

    def _create_fallback_logo(self) -> np.ndarray:
        """Create fallback logo if file not found."""
        # Create a 150x50 placeholder
        logo = np.zeros((50, 150, 3), dtype=np.uint8)
        logo.fill(40)  # Dark gray background

        # Add text
        cv2.putText(logo, "TH KOELN", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return logo

    def update_user_task(self, task: str):
        """Update the current user task."""
        self.current_user_task = task

    def add_robot_speech(self, speech: str, duration: float = 4.0):
        """Add robot speech to display queue."""
        self.speech_queue.append({"text": speech, "timestamp": time.time(), "duration": duration})

    def _get_active_speeches(self) -> list:
        """Get currently active speeches (within display duration)."""
        current_time = time.time()
        active = []

        for speech in self.speech_queue:
            elapsed = current_time - speech["timestamp"]
            if elapsed < speech["duration"]:
                active.append(speech["text"])

        return active

    def _wrap_text(self, text: str, max_width: int, font_scale: float) -> list:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            (text_width, _), _ = cv2.getTextSize(test_line, self.font, font_scale, self.thickness)

            if text_width <= max_width - 20:  # 20px margin
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def render_text_panel_sidebyside(self, frame: np.ndarray) -> np.ndarray:
        """
        Render text panel for side-by-side layout (below video).
        Creates 1280x720 output (1280x480 video + 1280x240 text panel).

        Args:
            frame: Input frame (1280x480 - two streams side by side)

        Returns:
            Combined frame (1280x720)
        """
        # Create text panel (1280x240)
        text_panel = np.zeros((240, 1280, 3), dtype=np.uint8)
        text_panel.fill(20)  # Dark background

        y_offset = 20

        # 1. User Task (persistent, top of panel)
        if self.current_user_task:
            cv2.putText(text_panel, "AUFGABE:", (10, y_offset), self.font, 0.5, (150, 150, 150), 1)
            y_offset += 25

            # Wrap task text
            task_lines = self._wrap_text(self.current_user_task, 1260, self.font_scale_task)

            for line in task_lines[:2]:  # Max 2 lines
                cv2.putText(text_panel, line, (10, y_offset), self.font, self.font_scale_task, self.color_task, self.thickness)
                y_offset += self.line_height

            y_offset += 10  # Spacing

        # 2. Robot Speech (timed, middle of panel)
        active_speeches = self._get_active_speeches()
        if active_speeches:
            cv2.putText(text_panel, "ROBOTER:", (10, y_offset), self.font, 0.5, (150, 150, 150), 1)
            y_offset += 25

            for speech in active_speeches[-2:]:  # Show last 2 speeches
                speech_lines = self._wrap_text(speech, 1260, self.font_scale_speech)

                for line in speech_lines[:2]:  # Max 2 lines per speech
                    cv2.putText(
                        text_panel, line, (10, y_offset), self.font, self.font_scale_speech, self.color_speech, self.thickness
                    )
                    y_offset += self.line_height

        # 3. Branding (bottom right)
        # Add logo
        logo_h, logo_w = self.logo.shape[:2]
        logo_x = 1280 - logo_w - 10
        logo_y = 240 - logo_h - 80
        text_panel[logo_y : logo_y + logo_h, logo_x : logo_x + logo_w] = self.logo

        # Add text below logo
        branding_lines = ["Prof. Dr. Daniel Gaida", "Labor für Physische KI", "TH Köln, Campus Gummersbach"]

        brand_y = logo_y + logo_h + 10
        for line in branding_lines:
            (text_width, text_height), _ = cv2.getTextSize(line, self.font, self.font_scale_branding, 1)

            text_x = 1280 - text_width - 10
            cv2.putText(text_panel, line, (text_x, brand_y), self.font, self.font_scale_branding, self.color_branding, 1)
            brand_y += 18

        # Combine video and text panel
        combined = np.vstack([frame, text_panel])

        return combined

    def render_text_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Render text overlay for overlay layout (on top of video).
        Keeps 640x480 resolution.

        Args:
            frame: Input frame (640x480)

        Returns:
            Frame with text overlay (640x480)
        """
        overlay = frame.copy()

        # 1. User Task (top, semi-transparent background)
        if self.current_user_task:
            # Background panel
            cv2.rectangle(overlay, (0, 0), (640, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, "AUFGABE:", (10, 20), self.font, 0.4, (150, 150, 150), 1)

            # Wrap and display task
            task_lines = self._wrap_text(self.current_user_task, 620, 0.45)

            y = 45
            for line in task_lines[:2]:
                cv2.putText(frame, line, (10, y), self.font, 0.45, self.color_task, 1)
                y += 18

        # 2. Robot Speech (middle-bottom, semi-transparent)
        active_speeches = self._get_active_speeches()
        if active_speeches:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 320), (640, 400), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, "ROBOTER:", (10, 340), self.font, 0.4, (150, 150, 150), 1)

            speech = active_speeches[-1]  # Show last speech only
            speech_lines = self._wrap_text(speech, 620, 0.45)

            y = 365
            for line in speech_lines[:2]:
                cv2.putText(frame, line, (10, y), self.font, 0.45, self.color_speech, 1)
                y += 18

        # 3. Branding (bottom right, semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (640 - 220, 480 - 90), (640, 480), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Smaller logo for overlay
        small_logo = cv2.resize(self.logo, (100, 33))
        logo_x = 640 - 110
        logo_y = 480 - 85
        frame[logo_y : logo_y + 33, logo_x : logo_x + 100] = small_logo

        # Branding text
        branding_lines = ["Prof. Dr. D. Gaida", "TH Köln"]

        y = 480 - 45
        for line in branding_lines:
            cv2.putText(frame, line, (640 - 210, y), self.font, 0.35, self.color_branding, 1)
            y += 16

        return frame


class EnhancedCameraRecorder:
    """Enhanced camera recorder with text overlays from Redis."""

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
        screenshot_dir: str = "screenshots",
        layout: str = "side-by-side",
    ):
        """Initialize the enhanced recorder."""
        self.camera_id = camera_id
        self.stream_name = stream_name
        self.fps = fps
        self.width = width
        self.height = height
        self.layout = layout
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(exist_ok=True)

        # Generate output filename
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"recording_{timestamp}.mp4"
        else:
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

        # Initialize Redis connections
        try:
            self.image_streamer = RedisImageStreamer(host=host, port=port, stream_name=stream_name)
            self.text_manager = RedisTextOverlayManager(host=host, port=port)
            print(f"✓ Connected to Redis at {host}:{port}")
        except Exception as e:
            self.camera.release()
            raise RuntimeError(f"Failed to connect to Redis: {e}")

        # Initialize overlay renderer
        if layout == "side-by-side":
            output_width = width * 2
            # output_height = 480
            final_width = output_width
            final_height = 720  # 480 + 240 for text panel
        else:  # overlay
            # output_width = width
            # output_height = height
            final_width = width
            final_height = height

        self.overlay_renderer = VideoOverlayRenderer(final_width, final_height, layout)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(self.output_file, fourcc, fps, (final_width, final_height))

        if not self.writer.isOpened():
            self.camera.release()
            raise RuntimeError("Failed to create video writer")

        print(f"✓ Video writer initialized: {self.output_file}")
        print(f"  Resolution: {final_width}x{final_height}")
        print(f"  Layout: {layout}")

        # State
        self.recording = True
        self.paused = False
        self.frame_count = 0
        self.start_time = time.time()
        self.last_annotated_frame = None

        # Start Redis subscriber thread
        self.text_update_thread = threading.Thread(target=self._subscribe_text_updates, daemon=True)
        self.text_update_thread.start()

        # Create window
        cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Recording", final_width, final_height)

        # Publish recording start message
        if self.text_manager:
            self.text_manager.publish_system_message("🎥 Aufnahme gestartet", duration_seconds=3.0)

    def _subscribe_text_updates(self):
        """Background thread to subscribe to text updates from Redis."""

        def on_text_update(text_data):
            text_type = text_data["type"]
            text = text_data["text"]

            if text_type == "user_task":
                self.overlay_renderer.update_user_task(text)
                print(f"📝 User task: {text}")

            elif text_type == "robot_speech":
                duration = text_data["metadata"].get("duration_seconds", 4.0)
                self.overlay_renderer.add_robot_speech(text, duration)
                print(f"🤖 Robot speech: {text}")

            elif text_type == "system_message":
                duration = text_data["metadata"].get("duration_seconds", 3.0)
                self.overlay_renderer.add_robot_speech(text, duration)
                print(f"ℹ️ System: {text}")

        try:
            self.text_manager.subscribe_to_texts(callback=on_text_update, block_ms=500)
        except Exception as e:
            print(f"Text subscription error: {e}")

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target dimensions."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            return cv2.resize(frame, (self.width, self.height))
        return frame

    def create_placeholder_frame(self) -> np.ndarray:
        """Create placeholder frame."""
        placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Waiting for annotated frames...",
            ((self.width - 400) // 2, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (128, 128, 128),
            2,
        )
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
            combined = np.hstack([camera_frame, annotated_frame])
        else:  # overlay
            combined = camera_frame.copy()
            overlay_width = self.width // 2
            overlay_height = self.height // 2
            overlay = cv2.resize(annotated_frame, (overlay_width, overlay_height))

            y_offset = self.height - overlay_height - 10
            x_offset = self.width - overlay_width - 10

            roi = combined[y_offset : y_offset + overlay_height, x_offset : x_offset + overlay_width]
            blended = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)
            combined[y_offset : y_offset + overlay_height, x_offset : x_offset + overlay_width] = blended

        return combined

    def run(self):
        """Main recording loop."""
        print("\n" + "=" * 60)
        print("Recording started - Press 'q' to stop")
        print("=" * 60 + "\n")

        no_frame_warning_shown = False

        try:
            while self.recording:
                # Read camera frame
                ret, camera_frame = self.camera.read()
                if not ret:
                    print("✗ Failed to read from camera")
                    break

                # Get annotated frame from Redis
                annotated_frame = None
                result = self.image_streamer.get_latest_image()

                if result:
                    annotated_frame, metadata = result
                    no_frame_warning_shown = False
                elif not no_frame_warning_shown:
                    print("⚠ No annotated frames available from Redis")
                    no_frame_warning_shown = True

                # Combine frames
                combined_frame = self.combine_frames(camera_frame, annotated_frame)

                # Apply text overlays
                if self.layout == "side-by-side":
                    display_frame = self.overlay_renderer.render_text_panel_sidebyside(combined_frame)
                else:
                    display_frame = self.overlay_renderer.render_text_overlay(combined_frame)

                # Write frame if not paused
                if not self.paused:
                    self.writer.write(display_frame)
                    self.frame_count += 1

                # Display
                cv2.imshow("Recording", display_frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    print("\nStopping recording...")
                    self.recording = False

                elif key == ord("p"):
                    self.paused = not self.paused
                    print(f"Recording {'paused' if self.paused else 'resumed'}")

                elif key == ord("s"):
                    self.save_screenshot(display_frame)

        except KeyboardInterrupt:
            print("\n\nRecording interrupted")

        finally:
            self.cleanup()

    def save_screenshot(self, frame: np.ndarray):
        """Save screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.screenshot_dir / f"screenshot_{timestamp}.png"

        try:
            cv2.imwrite(str(filename), frame)
            print(f"✓ Screenshot: {filename}")
        except Exception as e:
            print(f"✗ Screenshot failed: {e}")

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        self.camera.release()
        self.writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print("\n" + "=" * 60)
        print("RECORDING SUMMARY")
        print("=" * 60)
        print(f"  Output:      {self.output_file}")
        print(f"  Frames:      {self.frame_count}")
        print(f"  Duration:    {minutes:02d}:{seconds:02d}")
        print(f"  Avg FPS:     {self.frame_count / elapsed:.1f}")
        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced camera recorder with Redis text overlays")

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
    parser.add_argument("--screenshot-dir", type=str, default="screenshots")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ENHANCED CAMERA RECORDER WITH TEXT OVERLAYS")
    print("=" * 60)

    try:
        recorder = EnhancedCameraRecorder(
            camera_id=args.camera,
            stream_name=args.stream,
            host=args.host,
            port=args.port,
            output_file=args.output,
            fps=args.fps,
            width=args.width,
            height=args.height,
            codec=args.codec,
            screenshot_dir=args.screenshot_dir,
            layout=args.layout,
        )

        recorder.run()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
