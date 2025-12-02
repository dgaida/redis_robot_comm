#!/usr/bin/env python3
"""
record_camera_with_annotations.py

Records video from USB camera alongside annotated detection frames from Redis.
Creates a side-by-side video showing original camera feed and detection results.

Usage:
    python record_camera_with_annotations.py [options]

Examples:
    # Basic usage with default settings
    python record_camera_with_annotations.py

    # Custom camera and stream
    python record_camera_with_annotations.py --camera 1 --stream annotated_camera

    # Custom output file and frame rate
    python record_camera_with_annotations.py --output my_video.mp4 --fps 30

Controls (during recording):
    - Press 'q' to stop recording
    - Press 'p' to pause/unpause recording
    - Press 's' to take screenshot
"""

import cv2
import numpy as np
import argparse
import time
from datetime import datetime
from pathlib import Path
from redis_robot_comm import RedisImageStreamer


class CameraAnnotationRecorder:
    """Records USB camera feed with Redis annotated frames side-by-side."""

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
        """
        Initialize the recorder.

        Args:
            camera_id: USB camera device ID
            stream_name: Redis stream name for annotated images
            host: Redis server host
            port: Redis server port
            output_file: Output video file path (auto-generated if None)
            fps: Recording frame rate
            width: Camera frame width
            height: Camera frame height
            codec: Video codec (mp4v, XVID, H264, etc.)
            screenshot_dir: Directory for screenshots
            layout: 'side-by-side' or 'overlay'
        """
        self.camera_id = camera_id
        self.stream_name = stream_name
        self.fps = fps
        self.width = width
        self.height = height
        self.layout = layout
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(exist_ok=True)

        # Generate output filename if not provided
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

        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)

        print("✓ Camera opened successfully")

        # Initialize Redis streamer
        try:
            self.streamer = RedisImageStreamer(host=host, port=port, stream_name=stream_name)
            print(f"✓ Connected to Redis at {host}:{port}")
            print(f"✓ Listening to stream: {stream_name}")
        except Exception as e:
            self.camera.release()
            raise RuntimeError(f"Failed to connect to Redis: {e}")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)

        if layout == "side-by-side":
            output_width = width * 2
            output_height = height
        else:  # overlay
            output_width = width
            output_height = height

        self.writer = cv2.VideoWriter(self.output_file, fourcc, fps, (output_width, output_height))

        if not self.writer.isOpened():
            self.camera.release()
            raise RuntimeError(f"Failed to create video writer with codec {codec}")

        print(f"✓ Video writer initialized: {self.output_file}")
        print(f"  Resolution: {output_width}x{output_height}")
        print(f"  FPS: {fps}")
        print(f"  Layout: {layout}")

        # State variables
        self.recording = True
        self.paused = False
        self.frame_count = 0
        self.start_time = time.time()
        self.last_annotated_frame = None

        # Create display window
        cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Recording", output_width, output_height)

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to target dimensions.

        Args:
            frame: Input frame

        Returns:
            Resized frame
        """
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            return cv2.resize(frame, (self.width, self.height))
        return frame

    def create_placeholder_frame(self, text: str = "Waiting for annotated frames...") -> np.ndarray:
        """
        Create placeholder frame when annotated frame is not available.

        Args:
            text: Text to display

        Returns:
            Placeholder frame
        """
        placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Calculate text size for centering
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (self.width - text_width) // 2
        y = (self.height + text_height) // 2

        cv2.putText(placeholder, text, (x, y), font, font_scale, (128, 128, 128), thickness)

        return placeholder

    def combine_frames(self, camera_frame: np.ndarray, annotated_frame: np.ndarray = None) -> np.ndarray:
        """
        Combine camera and annotated frames based on layout.

        Args:
            camera_frame: Camera frame
            annotated_frame: Annotated frame (optional)

        Returns:
            Combined frame
        """
        # Ensure camera frame is correct size
        camera_frame = self.resize_frame(camera_frame)

        # Handle missing annotated frame
        if annotated_frame is None:
            if self.last_annotated_frame is not None:
                annotated_frame = self.last_annotated_frame
            else:
                annotated_frame = self.create_placeholder_frame()
        else:
            annotated_frame = self.resize_frame(annotated_frame)
            self.last_annotated_frame = annotated_frame.copy()

        if self.layout == "side-by-side":
            # Concatenate horizontally
            combined = np.hstack([camera_frame, annotated_frame])
        else:  # overlay
            # Place annotated frame on right side with transparency
            combined = camera_frame.copy()
            overlay_width = self.width // 2
            overlay_height = self.height // 2

            # Resize annotated frame for overlay
            overlay = cv2.resize(annotated_frame, (overlay_width, overlay_height))

            # Position in bottom-right corner
            y_offset = self.height - overlay_height - 10
            x_offset = self.width - overlay_width - 10

            # Add semi-transparent background
            roi = combined[y_offset : y_offset + overlay_height, x_offset : x_offset + overlay_width]
            blended = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)
            combined[y_offset : y_offset + overlay_height, x_offset : x_offset + overlay_width] = blended

        return combined

    def draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw recording information overlay.

        Args:
            frame: Input frame

        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Semi-transparent background panel
        panel_height = 90
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        line_height = 22
        x_offset = 10

        # Recording status
        y_pos = 20
        status = "PAUSED" if self.paused else "REC"
        status_color = (0, 165, 255) if self.paused else (0, 0, 255)
        cv2.circle(frame, (x_offset + 5, y_pos - 5), 6, status_color, -1)
        cv2.putText(frame, status, (x_offset + 20, y_pos), font, font_scale, color, thickness)

        # Frame count and time
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        text = f"Frames: {self.frame_count} | Time: {minutes:02d}:{seconds:02d}"
        cv2.putText(frame, text, (x_offset + 100, y_pos), font, font_scale, color, thickness)

        # Camera info
        y_pos += line_height
        text = f"Camera: {self.camera_id} | Stream: {self.stream_name}"
        cv2.putText(frame, text, (x_offset, y_pos), font, font_scale, color, thickness)

        # Output file
        y_pos += line_height
        text = f"Output: {self.output_file}"
        cv2.putText(frame, text, (x_offset, y_pos), font, font_scale, color, thickness)

        # Controls hint
        y_pos += line_height
        text = "Controls: Q=Quit | P=Pause | S=Screenshot"
        cv2.putText(frame, text, (x_offset, y_pos), font, 0.4, (200, 200, 200), thickness)

        return frame

    def save_screenshot(self, frame: np.ndarray):
        """
        Save current combined frame as screenshot.

        Args:
            frame: Frame to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.screenshot_dir / f"screenshot_{timestamp}.png"

        try:
            cv2.imwrite(str(filename), frame)
            print(f"✓ Screenshot saved: {filename}")
        except Exception as e:
            print(f"✗ Failed to save screenshot: {e}")

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

                # Get latest annotated frame from Redis
                annotated_frame = None
                result = self.streamer.get_latest_image()

                if result:
                    annotated_frame, metadata = result
                    no_frame_warning_shown = False
                elif not no_frame_warning_shown:
                    print("⚠ No annotated frames available from Redis stream")
                    print("  Using placeholder until frames arrive...")
                    no_frame_warning_shown = True

                # Combine frames
                combined_frame = self.combine_frames(camera_frame, annotated_frame)

                # Add info overlay
                display_frame = self.draw_info_overlay(combined_frame)

                # Write frame to video if not paused
                if not self.paused:
                    self.writer.write(combined_frame)
                    self.frame_count += 1

                # Display frame
                cv2.imshow("Recording", display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("\nStopping recording...")
                    self.recording = False

                elif key == ord("p"):
                    self.paused = not self.paused
                    status = "paused" if self.paused else "resumed"
                    print(f"Recording {status}")

                elif key == ord("s"):
                    self.save_screenshot(combined_frame)

        except KeyboardInterrupt:
            print("\n\nRecording interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        # Release resources
        self.camera.release()
        self.writer.release()
        cv2.destroyAllWindows()

        # Print summary
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print("\n" + "=" * 60)
        print("RECORDING SUMMARY")
        print("=" * 60)
        print(f"  Output file:    {self.output_file}")
        print(f"  Total frames:   {self.frame_count}")
        print(f"  Duration:       {minutes:02d}:{seconds:02d}")
        print(f"  Average FPS:    {self.frame_count / elapsed:.1f}")
        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Record USB camera with annotated detection frames from Redis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic recording
  python record_camera_with_annotations.py

  # Use camera 1 with custom stream
  python record_camera_with_annotations.py --camera 1 --stream my_annotations

  # Custom output file and frame rate
  python record_camera_with_annotations.py --output demo.mp4 --fps 60

  # Overlay layout instead of side-by-side
  python record_camera_with_annotations.py --layout overlay

Controls:
  q/ESC : Stop recording
  p     : Pause/unpause
  s     : Take screenshot
        """,
    )

    parser.add_argument("--camera", type=int, default=0, help="USB camera device ID (default: 0)")

    parser.add_argument(
        "--stream",
        type=str,
        default="annotated_camera",
        help="Redis stream name for annotated images (default: annotated_camera)",
    )

    parser.add_argument("--host", type=str, default="localhost", help="Redis server host (default: localhost)")

    parser.add_argument("--port", type=int, default=6379, help="Redis server port (default: 6379)")

    parser.add_argument(
        "--output", type=str, default=None, help="Output video file path (default: auto-generated with timestamp)"
    )

    parser.add_argument("--fps", type=int, default=30, help="Recording frame rate (default: 30)")

    parser.add_argument("--width", type=int, default=640, help="Camera frame width (default: 640)")

    parser.add_argument("--height", type=int, default=480, help="Camera frame height (default: 480)")

    parser.add_argument(
        "--codec", type=str, default="mp4v", choices=["mp4v", "XVID", "H264", "MJPG"], help="Video codec (default: mp4v)"
    )

    parser.add_argument(
        "--layout",
        type=str,
        default="side-by-side",
        choices=["side-by-side", "overlay"],
        help="Frame layout (default: side-by-side)",
    )

    parser.add_argument(
        "--screenshot-dir", type=str, default="screenshots", help="Directory for screenshots (default: screenshots)"
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "=" * 60)
    print("  CAMERA + ANNOTATION RECORDER")
    print("  USB Camera with Redis Annotated Frames")
    print("=" * 60)

    try:
        recorder = CameraAnnotationRecorder(
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
