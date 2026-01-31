#!/usr/bin/env python3
"""
visualize_annotated_frames.py

Real-time visualization of annotated frames from Redis stream.
Displays object detection results with bounding boxes, labels, and segmentation masks.

Usage:
    python visualize_annotated_frames.py [--stream-name STREAM_NAME] [--host HOST] [--port PORT]

Controls:
    - Press 'q' to quit
    - Press 's' to save current frame as screenshot
    - Press 'p' to pause/unpause
    - Press 'f' to toggle FPS display
"""

import cv2
import numpy as np
import argparse
import time
from datetime import datetime
from pathlib import Path
from redis_robot_comm import RedisImageStreamer


class AnnotatedFrameVisualizer:
    """Real-time visualizer for annotated detection frames from Redis."""

    def __init__(
        self,
        stream_name: str = "annotated_camera",
        host: str = "localhost",
        port: int = 6379,
        window_name: str = "Annotated Object Detection",
        fps_display: bool = True,
        save_directory: str = "screenshots",
    ):
        """
        Initialize the annotated frame visualizer.

        Args:
            stream_name: Redis stream name for annotated images
            host: Redis server host
            port: Redis server port
            window_name: OpenCV window name
            fps_display: Whether to display FPS counter
            save_directory: Directory to save screenshots
        """
        self.stream_name = stream_name
        self.window_name = window_name
        self.fps_display = fps_display
        self.save_directory = Path(save_directory)

        # Create save directory if it doesn't exist
        self.save_directory.mkdir(exist_ok=True)

        # Initialize Redis streamer
        try:
            self.streamer = RedisImageStreamer(host=host, port=port, stream_name=stream_name)
            print(f"✓ Connected to Redis at {host}:{port}")
            print(f"✓ Listening to stream: {stream_name}")
        except Exception as e:
            print(f"✗ Failed to connect to Redis: {e}")
            print("  Make sure Redis is running:")
            print("    docker run -p 6379:6379 redis:alpine")
            raise

        # State variables
        self.paused = False
        self.running = True
        self.frame_count = 0
        self.fps = 0.0
        self.last_frame_time = time.time()

        # Create OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 720)

    def update_fps(self):
        """Calculate and update FPS counter."""
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed > 0:
            self.fps = 1.0 / elapsed
        self.last_frame_time = current_time

    def draw_info_overlay(self, frame: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Draw information overlay on frame.

        Args:
            frame: Input frame
            metadata: Frame metadata from Redis

        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Semi-transparent background for info panel
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)
        line_height = 25
        x_offset = 10

        # Draw information lines
        y_pos = 25

        # Stream info
        text = f"Stream: {self.stream_name}"
        cv2.putText(frame, text, (x_offset, y_pos), font, font_scale, color, thickness)
        y_pos += line_height

        # Frame info
        frame_id = metadata.get("frame_id", "N/A")
        text = f"Frame: {frame_id} | Total Frames: {self.frame_count}"
        cv2.putText(frame, text, (x_offset, y_pos), font, font_scale, color, thickness)
        y_pos += line_height

        # Detection info
        detection_count = metadata.get("detection_count", 0)
        model_id = metadata.get("model_id", "unknown")
        text = f"Detections: {detection_count} | Model: {model_id}"
        cv2.putText(frame, text, (x_offset, y_pos), font, font_scale, color, thickness)
        y_pos += line_height

        # FPS counter
        if self.fps_display:
            text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, text, (x_offset, y_pos), font, font_scale, (0, 255, 0), thickness)

        # Status indicators
        status_x = width - 150
        if self.paused:
            cv2.putText(frame, "PAUSED", (status_x, 30), font, 0.7, (0, 165, 255), 2)

        return frame

    def save_screenshot(self, frame: np.ndarray):
        """
        Save current frame as screenshot.

        Args:
            frame: Frame to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.save_directory / f"annotated_frame_{timestamp}.png"

        try:
            cv2.imwrite(str(filename), frame)
            print(f"✓ Screenshot saved: {filename}")
        except Exception as e:
            print(f"✗ Failed to save screenshot: {e}")

    def handle_keyboard(self, key: int, frame: np.ndarray):
        """
        Handle keyboard input.

        Args:
            key: Key code from cv2.waitKey()
            frame: Current frame (for screenshot functionality)
        """
        if key == ord("q") or key == 27:  # 'q' or ESC
            self.running = False
            print("Quit requested")

        elif key == ord("s"):
            self.save_screenshot(frame)

        elif key == ord("p"):
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            print(f"Playback {status}")

        elif key == ord("f"):
            self.fps_display = not self.fps_display
            status = "enabled" if self.fps_display else "disabled"
            print(f"FPS display {status}")

        elif key == ord("h"):
            self.print_help()

    def print_help(self):
        """Print keyboard controls help."""
        print("\n" + "=" * 50)
        print("KEYBOARD CONTROLS")
        print("=" * 50)
        print("  q/ESC : Quit application")
        print("  s     : Save screenshot")
        print("  p     : Pause/unpause")
        print("  f     : Toggle FPS display")
        print("  h     : Show this help")
        print("=" * 50 + "\n")

    def run(self):
        """Main visualization loop."""
        print("\nStarting visualization...")
        print("Press 'h' for help\n")

        last_frame = None
        no_frame_count = 0
        max_no_frame_warnings = 5

        try:
            while self.running:
                # Get latest frame from Redis
                if not self.paused:
                    result = self.streamer.get_latest_image()

                    if result:
                        image, metadata = result
                        last_frame = image
                        no_frame_count = 0

                        # Update statistics
                        self.frame_count += 1
                        self.update_fps()

                        # Add info overlay
                        display_frame = self.draw_info_overlay(image, metadata)

                        # Display frame
                        cv2.imshow(self.window_name, display_frame)

                    else:
                        no_frame_count += 1

                        # Display last frame if available
                        if last_frame is not None:
                            cv2.imshow(self.window_name, last_frame)
                        else:
                            # Create placeholder frame
                            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(
                                placeholder,
                                "Waiting for frames...",
                                (150, 240),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (255, 255, 255),
                                2,
                            )
                            cv2.imshow(self.window_name, placeholder)

                        # Warn if no frames for extended period
                        if no_frame_count == 10 and no_frame_count <= max_no_frame_warnings:
                            print(f"⚠ No frames received from stream '{self.stream_name}'")
                            print("  Make sure the detection pipeline is running and publishing annotated frames")

                else:
                    # Paused - just display last frame
                    if last_frame is not None:
                        cv2.imshow(self.window_name, last_frame)

                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                if key != 255:  # 255 means no key pressed
                    self.handle_keyboard(key, last_frame if last_frame is not None else np.zeros((480, 640, 3)))

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        cv2.destroyAllWindows()
        print("✓ Visualization stopped")
        print(f"✓ Total frames displayed: {self.frame_count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize annotated object detection frames from Redis stream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default usage (localhost:6379, stream 'annotated_camera')
  python visualize_annotated_frames.py

  # Custom stream name
  python visualize_annotated_frames.py --stream-name my_annotated_stream

  # Custom Redis server
  python visualize_annotated_frames.py --host 192.168.1.100 --port 6380

  # Disable FPS display
  python visualize_annotated_frames.py --no-fps

Controls:
  q/ESC : Quit
  s     : Save screenshot
  p     : Pause/unpause
  f     : Toggle FPS display
  h     : Show help
        """,
    )

    parser.add_argument(
        "--stream-name",
        type=str,
        default="annotated_camera",
        help="Redis stream name for annotated images (default: annotated_camera)",
    )

    parser.add_argument("--host", type=str, default="localhost", help="Redis server host (default: localhost)")

    parser.add_argument("--port", type=int, default=6379, help="Redis server port (default: 6379)")

    parser.add_argument("--window-name", type=str, default="Annotated Object Detection", help="OpenCV window name")

    parser.add_argument("--no-fps", action="store_true", help="Disable FPS display")

    parser.add_argument(
        "--save-dir", type=str, default="screenshots", help="Directory to save screenshots (default: screenshots)"
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "=" * 60)
    print("  ANNOTATED FRAME VISUALIZER")
    print("  Real-time Object Detection Results from Redis")
    print("=" * 60)

    try:
        visualizer = AnnotatedFrameVisualizer(
            stream_name=args.stream_name,
            host=args.host,
            port=args.port,
            window_name=args.window_name,
            fps_display=not args.no_fps,
            save_directory=args.save_dir,
        )

        visualizer.run()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
