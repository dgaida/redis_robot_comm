#!/usr/bin/env python3
"""
record_camera_with_overlays.py

Enhanced recording script with proper Unicode/emoji support.
Now uses BaseVideoRecorder.
"""

import cv2
import numpy as np
import argparse
import time
import threading
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Optional, List

from redis_robot_comm import RedisTextOverlayManager
from scripts.recording.video_recorder_base import BaseVideoRecorder

# Try to import PIL for better text rendering
try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️ PIL not available - emoji support limited")


class VideoOverlayRenderer:
    """Handles rendering of text overlays with Unicode/emoji support."""

    def __init__(self, width: int, height: int, layout: str = "side-by-side"):
        """
        Initialize overlay renderer with Unicode support.

        Args:
            width: Video width
            height: Video height
            layout: 'side-by-side' or 'overlay'
        """
        self.width = width
        self.height = height
        self.layout = layout
        self.use_pil = HAS_PIL

        # Load logo with proper scaling (250px width)
        self.logo = self._load_logo()

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_task = 0.6
        self.font_scale_speech = 0.55
        self.font_scale_branding = 0.4
        self.thickness = 2
        self.line_height = 25

        # PIL fonts (if available) - support Unicode/emoji
        if self.use_pil:
            self._init_pil_fonts()

        # Colors
        self.color_task = (255, 255, 255)  # White
        self.color_speech = (100, 255, 100)  # Light green
        self.color_branding = (200, 200, 200)  # Light gray

        # Speech history
        self.speech_queue = deque(maxlen=5)
        self.current_user_task = None

    def _init_pil_fonts(self) -> None:
        """Initialize PIL fonts with Unicode/emoji support."""
        try:
            # Try to find a font with emoji support
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
                "C:\\Windows\\Fonts\\seguiemj.ttf",  # Windows emoji font
                "C:\\Windows\\Fonts\\arial.ttf",  # Windows fallback
            ]

            self.pil_font_task = None
            self.pil_font_speech = None
            self.pil_font_branding = None

            for font_path in font_paths:
                if Path(font_path).exists():
                    try:
                        self.pil_font_task = ImageFont.truetype(font_path, 24)
                        self.pil_font_speech = ImageFont.truetype(font_path, 22)
                        self.pil_font_branding = ImageFont.truetype(font_path, 16)
                        print(f"✓ Loaded Unicode font: {font_path}")
                        break
                    except Exception as e:
                        print(e)
                        continue

            if self.pil_font_task is None:
                # Fallback to default
                self.pil_font_task = ImageFont.load_default()
                self.pil_font_speech = ImageFont.load_default()
                self.pil_font_branding = ImageFont.load_default()
                print("⚠️ Using default font - emoji support may be limited")

        except Exception as e:
            print(f"⚠️ PIL font initialization failed: {e}")
            self.use_pil = False

    def _load_logo(self) -> np.ndarray:
        """Load and scale logo to 250px width keeping aspect ratio."""
        try:
            logo = cv2.imread("scripts/utils/thkoelnlogo.png")
            if logo is None:
                return self._create_fallback_logo()

            # Scale to 250px width keeping aspect ratio
            h, w = logo.shape[:2]
            target_width = 250
            aspect_ratio = h / w
            target_height = int(target_width * aspect_ratio)

            scaled_logo = cv2.resize(logo, (target_width, target_height))
            print(f"✓ Logo loaded and scaled to {target_width}x{target_height}")
            return scaled_logo

        except Exception as e:
            print(f"⚠️ Logo loading failed: {e}")
            return self._create_fallback_logo()

    def _create_fallback_logo(self) -> np.ndarray:
        """Create fallback logo (250px width)."""
        logo = np.zeros((83, 250, 3), dtype=np.uint8)
        logo.fill(40)

        # Add text using PIL if available for proper "Köln" rendering
        if self.use_pil:
            logo_pil = Image.fromarray(cv2.cvtColor(logo, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(logo_pil)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            except Exception:
                font = ImageFont.load_default()
            draw.text((20, 25), "TH KÖLN", fill=(255, 255, 255), font=font)
            logo = cv2.cvtColor(np.array(logo_pil), cv2.COLOR_RGB2BGR)
        else:
            # Fallback: use OpenCV (won't render ö correctly)
            cv2.putText(logo, "TH KOELN", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        return logo

    def update_user_task(self, task: str) -> None:
        """Update current user task."""
        self.current_user_task = task

    def add_robot_speech(self, speech: str, duration: float = 4.0) -> None:
        """Add robot speech to queue."""
        self.speech_queue.append({"text": speech, "timestamp": time.time(), "duration": duration})

    def _get_active_speeches(self) -> List[str]:
        """Get currently active speeches."""
        current_time = time.time()
        active = []

        for speech in self.speech_queue:
            elapsed = current_time - speech["timestamp"]
            if elapsed < speech["duration"]:
                active.append(speech["text"])

        return active

    def _put_text_pil(self, frame: np.ndarray, text: str, position: tuple, font, color: tuple) -> np.ndarray:
        """Render text using PIL (supports Unicode/emoji)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, text, font=font, fill=color)
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return frame_bgr

    def _wrap_text(self, text: str, max_width: int, font) -> List[str]:
        """Wrap text to fit width (PIL-aware)."""
        if not self.use_pil:
            # Fallback to simple word wrapping
            words = text.split()
            lines = []
            current_line = ""

            for word in words:
                test_line = f"{current_line} {word}".strip()
                if len(test_line) * 12 <= max_width - 20:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            return lines

        # PIL-based wrapping (accurate)
        words = text.split()
        lines = []
        current_line = ""

        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]

            if width <= max_width - 20:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def render_text_panel_sidebyside(self, frame: np.ndarray) -> np.ndarray:
        """Render text panel for side-by-side layout."""
        text_panel = np.zeros((240, 1280, 3), dtype=np.uint8)
        text_panel.fill(20)  # Dark background

        y_offset = 20

        if self.current_user_task:
            if self.use_pil:
                text_panel = self._put_text_pil(
                    text_panel, "AUFGABE:", (10, y_offset), self.pil_font_branding, (150, 150, 150)
                )
            else:
                cv2.putText(text_panel, "AUFGABE:", (10, y_offset), self.font, 0.5, (150, 150, 150), 1)
            y_offset += 25

            task_lines = self._wrap_text(self.current_user_task, 1260, self.pil_font_task if self.use_pil else None)

            for line in task_lines[:2]:
                if self.use_pil:
                    text_panel = self._put_text_pil(text_panel, line, (10, y_offset), self.pil_font_task, self.color_task)
                else:
                    cv2.putText(
                        text_panel, line, (10, y_offset), self.font, self.font_scale_task, self.color_task, self.thickness
                    )
                y_offset += self.line_height
            y_offset += 10

        active_speeches = self._get_active_speeches()
        if active_speeches:
            if self.use_pil:
                text_panel = self._put_text_pil(
                    text_panel, "ROBOTER:", (10, y_offset), self.pil_font_branding, (150, 150, 150)
                )
            else:
                cv2.putText(text_panel, "ROBOTER:", (10, y_offset), self.font, 0.5, (150, 150, 150), 1)
            y_offset += 25

            for speech in active_speeches[-2:]:
                speech_lines = self._wrap_text(speech, 1260, self.pil_font_speech if self.use_pil else None)

                for line in speech_lines[:2]:
                    if self.use_pil:
                        text_panel = self._put_text_pil(
                            text_panel, line, (10, y_offset), self.pil_font_speech, self.color_speech
                        )
                    else:
                        clean_line = "".join(c for c in line if ord(c) < 128)
                        cv2.putText(
                            text_panel,
                            clean_line,
                            (10, y_offset),
                            self.font,
                            self.font_scale_speech,
                            self.color_speech,
                            self.thickness,
                        )
                    y_offset += self.line_height

        logo_h, logo_w = self.logo.shape[:2]
        logo_x = 1280 - logo_w - 10
        logo_y = 240 - logo_h - 80
        text_panel[logo_y : logo_y + logo_h, logo_x : logo_x + logo_w] = self.logo

        branding_lines = ["Prof. Dr. Daniel Gaida", "Labor für Physische KI", "TH Köln, Campus Gummersbach"]

        brand_y = logo_y + logo_h + 10
        for line in branding_lines:
            if self.use_pil:
                draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
                bbox = draw.textbbox((0, 0), line, font=self.pil_font_branding)
                text_width = bbox[2] - bbox[0]
                text_x = 1280 - text_width - 10
                text_panel = self._put_text_pil(
                    text_panel, line, (text_x, brand_y), self.pil_font_branding, self.color_branding
                )
            else:
                clean_line = line.replace("ö", "o")
                (text_width, _), _ = cv2.getTextSize(clean_line, self.font, self.font_scale_branding, 1)
                text_x = 1280 - text_width - 10
                cv2.putText(
                    text_panel, clean_line, (text_x, brand_y), self.font, self.font_scale_branding, self.color_branding, 1
                )
            brand_y += 18

        combined = np.vstack([frame, text_panel])
        return combined

    def render_text_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Render text overlay for overlay layout."""
        overlay = frame.copy()

        if self.current_user_task:
            cv2.rectangle(overlay, (0, 0), (640, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            if self.use_pil:
                frame = self._put_text_pil(frame, "AUFGABE:", (10, 20), self.pil_font_branding, (150, 150, 150))
            else:
                cv2.putText(frame, "AUFGABE:", (10, 20), self.font, 0.4, (150, 150, 150), 1)

            task_lines = self._wrap_text(self.current_user_task, 620, self.pil_font_task if self.use_pil else None)

            y = 45
            for line in task_lines[:2]:
                if self.use_pil:
                    frame = self._put_text_pil(frame, line, (10, y), self.pil_font_task, self.color_task)
                else:
                    cv2.putText(frame, line, (10, y), self.font, 0.45, self.color_task, 1)
                y += 18

        active_speeches = self._get_active_speeches()
        if active_speeches:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 320), (640, 400), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            if self.use_pil:
                frame = self._put_text_pil(frame, "ROBOTER:", (10, 340), self.pil_font_branding, (150, 150, 150))
            else:
                cv2.putText(frame, "ROBOTER:", (10, 340), self.font, 0.4, (150, 150, 150), 1)

            speech = active_speeches[-1]
            speech_lines = self._wrap_text(speech, 620, self.pil_font_speech if self.use_pil else None)

            y = 365
            for line in speech_lines[:2]:
                if self.use_pil:
                    frame = self._put_text_pil(frame, line, (10, y), self.pil_font_speech, self.color_speech)
                else:
                    clean_line = "".join(c for c in line if ord(c) < 128)
                    cv2.putText(frame, clean_line, (10, y), self.font, 0.45, self.color_speech, 1)
                y += 18

        overlay = frame.copy()
        cv2.rectangle(overlay, (640 - 280, 480 - 120), (640, 480), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        logo_h, logo_w = self.logo.shape[:2]
        small_w = 166
        small_h = int(logo_h * (small_w / logo_w))
        small_logo = cv2.resize(self.logo, (small_w, small_h))

        logo_x = 640 - small_w - 10
        logo_y = 480 - 115
        frame[logo_y : logo_y + small_h, logo_x : logo_x + small_w] = small_logo

        branding_lines = ["Prof. Dr. D. Gaida", "TH Köln"]
        y = 480 - 50
        for line in branding_lines:
            if self.use_pil:
                frame = self._put_text_pil(frame, line, (640 - 270, y), self.pil_font_branding, self.color_branding)
            else:
                clean_line = line.replace("ö", "o")
                cv2.putText(frame, clean_line, (640 - 270, y), self.font, 0.35, self.color_branding, 1)
            y += 16

        return frame


class EnhancedCameraRecorder(BaseVideoRecorder):
    """Enhanced camera recorder with Unicode text overlays."""

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
        screenshot_dir: str = "screenshots",
        layout: str = "side-by-side",
    ):
        super().__init__(
            camera_id=camera_id,
            stream_name=stream_name,
            host=host,
            port=port,
            fps=fps,
            width=width,
            height=height,
            output_file=output_file,
            codec=codec,
        )
        self.layout = layout
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(exist_ok=True)

        try:
            self.text_manager = RedisTextOverlayManager(host=host, port=port)
        except Exception as e:
            self.camera.release()
            raise RuntimeError(f"Failed to connect to Redis Text Manager: {e}")

        if layout == "side-by-side":
            output_width = width * 2
            final_width = output_width
            final_height = 720
        else:
            final_width = width
            final_height = height

        self.overlay_renderer = VideoOverlayRenderer(final_width, final_height, layout)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(self.output_file, fourcc, fps, (final_width, final_height))

        if not self.writer.isOpened():
            self.camera.release()
            raise RuntimeError("Failed to create video writer")

        self.text_update_thread = threading.Thread(target=self._subscribe_text_updates, daemon=True)
        self.text_update_thread.start()

        cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Recording", final_width, final_height)

        if self.text_manager:
            self.text_manager.publish_system_message("🎥 Aufnahme gestartet", duration_seconds=3.0)

    def _subscribe_text_updates(self) -> None:
        """Background thread for text updates."""
        def on_text_update(text_data):
            text_type = text_data["type"]
            text = text_data["text"]

            if text_type == "user_task":
                self.overlay_renderer.update_user_task(text)
            elif text_type in ("robot_speech", "system_message"):
                duration = text_data["metadata"].get("duration_seconds", 4.0)
                self.overlay_renderer.add_robot_speech(text, duration)

        try:
            self.text_manager.subscribe_to_texts(callback=on_text_update, block_ms=500)
        except Exception as e:
            print(f"Text subscription error: {e}")

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
            display_frame = self.overlay_renderer.render_text_panel_sidebyside(combined)
        else:
            # overlay
            combined = camera_frame.copy()
            overlay_width = self.width // 2
            overlay_height = self.height // 2
            overlay = cv2.resize(annotated_frame, (overlay_width, overlay_height))

            y_offset = self.height - overlay_height - 10
            x_offset = self.width - overlay_width - 10
            roi = combined[y_offset : y_offset + overlay_height, x_offset : x_offset + overlay_width]
            blended = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)
            combined[y_offset : y_offset + overlay_height, x_offset : x_offset + overlay_width] = blended
            display_frame = self.overlay_renderer.render_text_overlay(combined)

        return display_frame

    def run(self) -> None:
        """Main recording loop."""
        try:
            while self.recording:
                ret, camera_frame = self.camera.read()
                if not ret:
                    break

                result = self.image_streamer.get_latest_image()
                annotated_frame = result[0] if result else None

                display_frame = self.process_frame(camera_frame, annotated_frame)

                if not self.paused:
                    self.writer.write(display_frame)
                    self.frame_count += 1

                cv2.imshow("Recording", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    self.recording = False
                elif key == ord("p"):
                    self.paused = not self.paused
                elif key == ord("s"):
                    self.save_screenshot(display_frame)

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def save_screenshot(self, frame: np.ndarray) -> None:
        """Save screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.screenshot_dir / f"screenshot_{timestamp}.png"
        cv2.imwrite(str(filename), frame)

    def cleanup(self) -> None:
        """Clean up resources."""
        self.writer.release()
        super().cleanup()


def main():
    parser = argparse.ArgumentParser(description="Enhanced camera recorder")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--stream", type=str, default="annotated_camera")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--layout", type=str, default="side-by-side", choices=["side-by-side", "overlay"])
    args = parser.parse_args()

    recorder = EnhancedCameraRecorder(
        camera_id=args.camera,
        stream_name=args.stream,
        host=args.host,
        port=args.port,
        output_file=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        layout=args.layout,
    )
    recorder.run()

if __name__ == "__main__":
    main()
