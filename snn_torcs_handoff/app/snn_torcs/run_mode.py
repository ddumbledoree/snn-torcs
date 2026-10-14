"""
Run-mode utilities for TORCS training/evaluation dual-mode system.

Provides centralized control over:
- Xvfb virtual display management (fast_train always uses :99)
- TORCS XML config backup/patch/restore (practice.xml, screen.xml)
- Launch parameters (env vars, sleep times, output suppression)

Two modes:
  fast_train — headless, minimal rendering, for long training runs
  demo_eval  — full GUI, for third-party evaluation and acceptance

When run_mode is None, all functions return safe defaults that preserve
the exact current behavior (legacy path).
"""
import os
import re
import shutil
import subprocess
import time

VALID_MODES = ("fast_train", "demo_eval")
XVFB_DISPLAY = ":99"

PRACTICE_XML = os.path.expanduser("~/.torcs/config/raceman/practice.xml")
SCREEN_XML = os.path.expanduser("~/.torcs/config/screen.xml")
BACKUP_PRACTICE_XML = PRACTICE_XML + ".bak"
BACKUP_SCREEN_XML = SCREEN_XML + ".bak"

# Module-level state
_xvfb_proc = None
_configs_backed_up = False


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
def validate_mode(mode):
    """Return mode if valid, else raise ValueError."""
    if mode is not None and mode not in VALID_MODES:
        raise ValueError(
            f"Invalid run_mode '{mode}'. Must be one of {VALID_MODES} or None.")
    return mode


# ------------------------------------------------------------------
# Display management
# ------------------------------------------------------------------
def _xvfb_is_running():
    """Check if Xvfb is already active on XVFB_DISPLAY."""
    try:
        result = subprocess.run(
            ["xdpyinfo", "-display", XVFB_DISPLAY],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=2)
        return result.returncode == 0
    except Exception:
        return False


def ensure_display(mode):
    """Ensure the correct display is available for the given mode.

    fast_train: always use Xvfb :99 (never the real DISPLAY).
    demo_eval:  no-op (caller checks DISPLAY before invocation).
    """
    global _xvfb_proc

    if mode == "demo_eval":
        return  # requires real DISPLAY; caller validates

    if mode != "fast_train":
        return  # None / legacy — no display management

    # fast_train: always force Xvfb :99
    if _xvfb_is_running():
        os.environ["DISPLAY"] = XVFB_DISPLAY
        return

    # Start Xvfb
    try:
        _xvfb_proc = subprocess.Popen(
            ["Xvfb", XVFB_DISPLAY, "-screen", "0", "640x480x24", "-ac"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError(
            "Xvfb is required for headless fast_train mode but not found. "
            "Install with: sudo apt install xvfb")

    # Wait for Xvfb to become ready
    for _ in range(20):
        if _xvfb_is_running():
            break
        time.sleep(0.25)
    else:
        raise RuntimeError("Xvfb failed to start on %s" % XVFB_DISPLAY)

    os.environ["DISPLAY"] = XVFB_DISPLAY


def stop_xvfb():
    """Terminate Xvfb if we started it. Never kills pre-existing instances."""
    global _xvfb_proc
    if _xvfb_proc is not None:
        try:
            _xvfb_proc.terminate()
            _xvfb_proc.wait(timeout=5)
        except Exception:
            try:
                _xvfb_proc.kill()
            except Exception:
                pass
        _xvfb_proc = None


# ------------------------------------------------------------------
# XML config backup / patch / restore
# ------------------------------------------------------------------
def backup_configs():
    """One-shot backup of practice.xml and screen.xml.

    Only backs up once per session (idempotent). Preserves the user's
    original settings regardless of what they were.
    """
    global _configs_backed_up
    if _configs_backed_up:
        return

    for src, bak in [(PRACTICE_XML, BACKUP_PRACTICE_XML),
                     (SCREEN_XML, BACKUP_SCREEN_XML)]:
        if os.path.isfile(src) and not os.path.isfile(bak):
            shutil.copy2(src, bak)

    _configs_backed_up = True


def set_display_mode(mode):
    """Patch practice.xml display mode attribute.

    fast_train: 'results only' (RM_DISP_MODE_NONE — pure physics, no 3D)
    demo_eval:  'normal'       (full OpenGL rendering — forced to recover
                                 from prior crash leaving stale config)
    """
    if mode is None:
        return

    if not os.path.isfile(PRACTICE_XML):
        return

    new_val = "results only" if mode == "fast_train" else "normal"
    _patch_xml_attr(PRACTICE_XML, "display mode", new_val)


def set_screen_resolution(mode):
    """Patch screen.xml resolution.

    fast_train: 64x64 @ 16bpp (minimal rendering surface)
    demo_eval:  restore from backup (never hardcode a specific resolution)
    """
    if mode is None:
        return

    if mode == "fast_train":
        _patch_xml_attr(SCREEN_XML, "x", "64")
        _patch_xml_attr(SCREEN_XML, "y", "64")
        _patch_xml_attr(SCREEN_XML, "bpp", "16")
    elif mode == "demo_eval":
        # Restore original from backup if available
        if os.path.isfile(BACKUP_SCREEN_XML):
            shutil.copy2(BACKUP_SCREEN_XML, SCREEN_XML)


def restore_configs():
    """Restore practice.xml and screen.xml from .bak files.

    Called during end() cleanup. Deletes .bak files after restore.
    """
    global _configs_backed_up

    for bak, dst in [(BACKUP_PRACTICE_XML, PRACTICE_XML),
                     (BACKUP_SCREEN_XML, SCREEN_XML)]:
        if os.path.isfile(bak):
            shutil.copy2(bak, dst)
            try:
                os.remove(bak)
            except OSError:
                pass

    _configs_backed_up = False


def _patch_xml_attr(path, attr_name, new_val):
    """Replace an XML attribute value in-place using regex.

    Safe for TORCS XML — preserves all whitespace and other attributes.
    """
    if not os.path.isfile(path):
        return

    with open(path, "r") as f:
        content = f.read()

    # Match both <attstr name="X" val="Y"/> and <attnum name="X" val="Y"/>
    pattern = r'(<(?:attstr|attnum) name="' + re.escape(attr_name) + r'" val=")([^"]*)(")'
    new_content, count = re.subn(pattern, r'\g<1>' + new_val + r'\g<3>', content)

    if count > 0:
        with open(path, "w") as f:
            f.write(new_content)


# ------------------------------------------------------------------
# Launch parameter helpers
# ------------------------------------------------------------------
def get_launch_env(mode):
    """Return extra env vars for TORCS subprocess."""
    if mode == "fast_train":
        return {"DISPLAY": XVFB_DISPLAY}
    return {}  # demo_eval / None: inherit current environment


def get_launch_sleep(mode):
    """Return sleep duration (seconds) after launching TORCS."""
    if mode == "fast_train":
        return 0.3
    return 0.5  # demo_eval / None: current behavior


def should_suppress_output(mode):
    """Whether to redirect TORCS stdout/stderr to DEVNULL."""
    return mode == "fast_train"
