import ctypes
import os
import sys
from contextlib import contextmanager

import numpy as np

if os.name == "nt":  # Windows
    libc = ctypes.CDLL("msvcrt.dll")
    _controlfp = libc._controlfp
    _controlfp.argtypes = [ctypes.c_uint, ctypes.c_uint]
    _controlfp.restype = ctypes.c_uint

    def save_fenv():
        """Save the current floating-point environment."""
        return _controlfp(0, 0)

    def restore_fenv(env):
        """Restore the saved floating-point environment."""
        _controlfp(env, 0xFFFFFFFF)

elif sys.platform.startswith("linux"):  # Linux
    libc = ctypes.CDLL("libm.so.6")

    # Define floating-point environment type (fenv_t)
    class FEnv(ctypes.Structure):
        _fields_ = [("data", ctypes.c_uint8 * 28)]  # Adjust size for your architecture

    fegetenv = libc.fegetenv
    fegetenv.argtypes = [ctypes.POINTER(FEnv)]
    fegetenv.restype = ctypes.c_int

    fesetenv = libc.fesetenv
    fesetenv.argtypes = [ctypes.POINTER(FEnv)]
    fesetenv.restype = ctypes.c_int

    def save_fenv():
        """Save the current floating-point environment."""
        env = FEnv()
        if fegetenv(ctypes.byref(env)) != 0:
            raise RuntimeError("Failed to save floating-point environment.")
        return env

    def restore_fenv(env):
        """Restore the saved floating-point environment."""
        if fesetenv(ctypes.byref(env)) != 0:
            raise RuntimeError("Failed to restore floating-point environment.")

else:
    raise NotImplementedError("Unsupported platform")

def print_fenv_state(stage):
    """Print the floating-point environment state for debugging."""
    try:
        state = save_fenv()
        if isinstance(state, int):
            print(f"[{stage}] Floating-Point Control Word: {state:#010x}")
        else:
            print(f"[{stage}] Floating-Point Environment: {bytes(state.data).hex()}")
    except Exception as e:
        print(f"[{stage}] Error: {e}")


@contextmanager
def fpenv_context_restore(name="", verbose=True):
    """
    Wrapper to restore floating-point exception state
    after `init()` from Framsticks DLL.

    This is important to avoid floating point exceptions and others.
    
    ## Usage
    ```
    with fpenv_context_restore():
        framsLib = ...
    ```

    ## See also
    [numpy.seterr function](https://numpy.org/doc/stable/reference/generated/numpy.seterr.html)

    ```
    orig_settings = np.seterr(all='ignore')
    ```
    """
    original_control_word = save_fenv()
    if verbose:
        print_fenv_state(f"Before {name}")
    try:
        yield
    finally:
        if verbose:
            print_fenv_state(f"After {name}")
        restore_fenv(original_control_word)

