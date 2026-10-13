import mss
import numpy as np
import cv2
import time

class ScreenCapture:
    """
    Captures frames from the screen or a specified monitor.
    """
    def __init__(self, monitor_number: int = 1):
        """
        Initializes the screen capture with a specific monitor.
        Args:
            monitor_number (int): The index of the monitor to capture (1 for primary, 2 for secondary, etc.).
                                  Default is 1 (primary).
        """
        self.sct = mss.mss()
        self.monitor_number = monitor_number
        self.monitor = self._get_monitor_info(monitor_number)
        if self.monitor is None:
            raise ValueError(f"Monitor {monitor_number} not found. Available monitors: {self.sct.monitors}")

    def _get_monitor_info(self, monitor_index):
        """Helper to get monitor geometry."""
        if 0 < monitor_index <= len(self.sct.monitors):
            return self.sct.monitors[monitor_index]
        return None

    def grab_frame(self) -> np.ndarray:
        """
        Grabs a single frame from the specified monitor.
        Returns:
            np.ndarray: A NumPy array representing the captured image (BGR format),
                        or None if capture fails.
        """
        try:
            sct_img = self.sct.grab(self.monitor)
            # Convert to a NumPy array
            img = np.array(sct_img)
            # The 'mss' library captures in BGRA format. Convert to BGR for OpenCV.
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except mss.exception.ScreenShotError as e:
            print(f"Screen capture error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during screen capture: {e}")
            return None

    def __repr__(self):
        return f"ScreenCapture(monitor_number={self.monitor_number}, monitor_geometry={self.monitor})"
