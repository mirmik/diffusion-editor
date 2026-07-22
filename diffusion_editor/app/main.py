"""Entry point for diffusion-editor.

Runs on ``termin.display.SDLBackendWindow`` — backend is selected by
the env-var ``TERMIN_BACKEND`` (``opengl`` / ``vulkan``). A single
tgfx2 IRenderDevice is owned by the window; the UI borrows it via the
shared ``Tgfx2Context`` so every renderer lands on the same device.
"""

import sys
import ctypes
import sdl2
from sdl2 import video

from tcbase import Key, MouseButton, Mods
from tcbase import log

from tgfx import Tgfx2Context, configure_default_shader_runtime
from termin.display import SDLBackendWindow

from .editor_window import EditorWindow


_KEY_MAP = {
    sdl2.SDL_SCANCODE_BACKSPACE: Key.BACKSPACE,
    sdl2.SDL_SCANCODE_DELETE: Key.DELETE,
    sdl2.SDL_SCANCODE_LEFT: Key.LEFT,
    sdl2.SDL_SCANCODE_RIGHT: Key.RIGHT,
    sdl2.SDL_SCANCODE_UP: Key.UP,
    sdl2.SDL_SCANCODE_DOWN: Key.DOWN,
    sdl2.SDL_SCANCODE_HOME: Key.HOME,
    sdl2.SDL_SCANCODE_END: Key.END,
    sdl2.SDL_SCANCODE_RETURN: Key.ENTER,
    sdl2.SDL_SCANCODE_ESCAPE: Key.ESCAPE,
    sdl2.SDL_SCANCODE_TAB: Key.TAB,
    sdl2.SDL_SCANCODE_SPACE: Key.SPACE,
}


def translate_key(scancode: int) -> int:
    if scancode in _KEY_MAP:
        return _KEY_MAP[scancode]

    # Letters: SDL scancodes are layout-independent; map directly to A..Z.
    if sdl2.SDL_SCANCODE_A <= scancode <= sdl2.SDL_SCANCODE_Z:
        return Key(Key.A + (scancode - sdl2.SDL_SCANCODE_A))

    # Digits row.
    digit_map = {
        sdl2.SDL_SCANCODE_0: Key.KEY_0,
        sdl2.SDL_SCANCODE_1: Key.KEY_1,
        sdl2.SDL_SCANCODE_2: Key.KEY_2,
        sdl2.SDL_SCANCODE_3: Key.KEY_3,
        sdl2.SDL_SCANCODE_4: Key.KEY_4,
        sdl2.SDL_SCANCODE_5: Key.KEY_5,
        sdl2.SDL_SCANCODE_6: Key.KEY_6,
        sdl2.SDL_SCANCODE_7: Key.KEY_7,
        sdl2.SDL_SCANCODE_8: Key.KEY_8,
        sdl2.SDL_SCANCODE_9: Key.KEY_9,
    }
    if scancode in digit_map:
        return digit_map[scancode]

    keycode = sdl2.SDL_GetKeyFromScancode(scancode)
    # SDL keycodes for letters are lowercase ASCII; normalize to uppercase.
    if 97 <= keycode <= 122:
        keycode -= 32
    if 0 <= keycode < 128:
        try:
            return Key(keycode)
        except ValueError:
            pass
    return Key.UNKNOWN


def translate_mods(sdl_mods: int) -> int:
    result = 0
    if sdl_mods & (sdl2.KMOD_LSHIFT | sdl2.KMOD_RSHIFT):
        result |= Mods.SHIFT.value
    if sdl_mods & (sdl2.KMOD_LCTRL | sdl2.KMOD_RCTRL):
        result |= Mods.CTRL.value
    if sdl_mods & (sdl2.KMOD_LALT | sdl2.KMOD_RALT):
        result |= Mods.ALT.value
    return result


_SDL_BUTTON_MAP = {
    1: MouseButton.LEFT,
    2: MouseButton.MIDDLE,
    3: MouseButton.RIGHT,
}


def translate_button(sdl_button: int) -> MouseButton:
    return _SDL_BUTTON_MAP.get(sdl_button, MouseButton.LEFT)


# --- Cursor ---

_cursor_cache: dict[str, object] = {}

def _set_sdl_cursor(name: str):
    if name in _cursor_cache:
        sdl2.SDL_SetCursor(_cursor_cache[name])
        return
    cursor_map = {
        "arrow": sdl2.SDL_SYSTEM_CURSOR_ARROW,
        "cross": sdl2.SDL_SYSTEM_CURSOR_CROSSHAIR,
        "hand": sdl2.SDL_SYSTEM_CURSOR_HAND,
        "text": sdl2.SDL_SYSTEM_CURSOR_IBEAM,
        "move": sdl2.SDL_SYSTEM_CURSOR_SIZEALL,
    }
    sys_id = cursor_map.get(name, sdl2.SDL_SYSTEM_CURSOR_ARROW)
    cursor = sdl2.SDL_CreateSystemCursor(sys_id)
    _cursor_cache[name] = cursor
    sdl2.SDL_SetCursor(cursor)


# --- Main ---

def main():
    import faulthandler
    faulthandler.enable()
    log.set_level(log.Level.INFO)

    if not configure_default_shader_runtime("diffusion-editor"):
        raise RuntimeError(
            "Termin shader runtime is unavailable. Check that TERMIN_SDK points "
            "to the installed SDK and contains bin/termin_shaderc and bin/slangc."
        )

    # SDLBackendWindow owns the device + SDL window; backend picked from
    # TERMIN_BACKEND env-var.
    window = SDLBackendWindow("Diffusion Editor", 1280, 800)
    tgfx2_ctx = Tgfx2Context.from_window(window.device_ptr(), window.context_ptr())
    log.info("[main] Window created")

    editor = EditorWindow(ctx=tgfx2_ctx)
    ui = editor.ui

    # Cursor support
    def on_cursor_changed(cursor_name: str):
        if cursor_name:
            _set_sdl_cursor(cursor_name)
        else:
            _set_sdl_cursor("arrow")
    ui.on_cursor_changed = on_cursor_changed

    # Handle command line arguments
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if path.lower().endswith(".deproj"):
            editor.open_file_path(path)
        else:
            editor.import_image_path(path)

    sdl2.SDL_StartTextInput()
    event = sdl2.SDL_Event()

    def dispatch(ev):
        t = ev.type
        if t == sdl2.SDL_QUIT:
            editor.request_stop()
            window.set_should_close(True)
        elif t == sdl2.SDL_WINDOWEVENT:
            if ev.window.event == video.SDL_WINDOWEVENT_CLOSE:
                editor.request_stop()
                window.set_should_close(True)
        elif t == sdl2.SDL_MOUSEMOTION:
            ui.mouse_move(float(ev.motion.x), float(ev.motion.y))
        elif t == sdl2.SDL_MOUSEBUTTONDOWN:
            ui.mouse_down(float(ev.button.x), float(ev.button.y),
                          translate_button(ev.button.button),
                          translate_mods(sdl2.SDL_GetModState()))
        elif t == sdl2.SDL_MOUSEBUTTONUP:
            ui.mouse_up(float(ev.button.x), float(ev.button.y))
        elif t == sdl2.SDL_MOUSEWHEEL:
            mx, my = ctypes.c_int(), ctypes.c_int()
            sdl2.SDL_GetMouseState(ctypes.byref(mx), ctypes.byref(my))
            ui.mouse_wheel(float(ev.wheel.x), float(ev.wheel.y),
                           float(mx.value), float(my.value))
        elif t == sdl2.SDL_KEYDOWN:
            key = translate_key(ev.key.keysym.scancode)
            mods = translate_mods(sdl2.SDL_GetModState())
            ui.key_down(key, mods)
        elif t == sdl2.SDL_TEXTINPUT:
            ui.text_input(ev.text.text.decode("utf-8"))

    try:
        while editor.running:
            if sdl2.SDL_WaitEventTimeout(ctypes.byref(event), 50):
                dispatch(event)
                while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
                    dispatch(event)

            ui.process_deferred()

            if not editor.running:
                break

            # Poll engines
            editor.poll()

            # Render into the UI's offscreen texture, then publish via
            # SDLBackendWindow.present — on OpenGL this blits to the
            # default framebuffer and calls SwapWindow; on Vulkan it
            # acquires a swapchain image, blits, submits, presents.
            vw, vh = window.framebuffer_size()
            tex = editor.render_compose(vw, vh)
            if tex is not None:
                window.present(tex)
    finally:
        # Application shutdown stops view workers, inference workers and GPU
        # resources before the borrowed host/window leaves scope.
        editor.close()
        # SDLBackendWindow destructor cleans up SDL window + GL/Vulkan
        # device in the correct order. Dropping the Python wrapper is
        # enough; explicit sdl2.SDL_Quit is handled by SDLBackendWindow.


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
