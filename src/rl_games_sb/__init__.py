import os
import sys

# On macOS, OpenCV (imported by SB3) and pygame (used by the Box2D envs) each
# bundle their own SDL2, and loading both makes the Objective-C runtime print
# ~17 "Class SDL... is implemented in both" warnings. They are harmless here:
# nothing uses OpenCV's SDL. Load both once with stderr silenced.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
if sys.platform == "darwin":
    _stderr = os.dup(2)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    try:
        import cv2  # noqa: F401
        import pygame  # noqa: F401
    finally:
        os.dup2(_stderr, 2)
        os.close(_devnull)
        os.close(_stderr)
