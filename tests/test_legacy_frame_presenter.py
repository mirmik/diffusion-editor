from diffusion_editor.app.legacy_main import LegacyFramePresenter
from diffusion_editor.app.editor_window import _LegacyPresentation


class _Editor:
    def __init__(self):
        self.repaint_requested = True
        self.worker_ready = False
        self.poll_calls = 0
        self.render_calls = []

    def poll(self):
        self.poll_calls += 1
        if self.worker_ready:
            self.worker_ready = False
            self.request_repaint()

    def request_repaint(self):
        self.repaint_requested = True

    def consume_repaint_request(self):
        requested = self.repaint_requested
        self.repaint_requested = False
        return requested

    def render_compose(self, width, height):
        self.render_calls.append((width, height))
        return f"frame-{len(self.render_calls)}"


class _Window:
    def __init__(self):
        self.size = (1280, 800)
        self.presented = []

    def framebuffer_size(self):
        return self.size

    def present(self, texture):
        self.presented.append(texture)


def test_legacy_presenter_skips_idle_but_keeps_polling():
    editor = _Editor()
    window = _Window()
    presenter = LegacyFramePresenter(editor, window)

    assert presenter.tick() is True
    assert presenter.tick() is False

    assert editor.poll_calls == 2
    assert editor.render_calls == [(1280, 800)]
    assert window.presented == ["frame-1"]


def test_legacy_presenter_coalesces_input_and_worker_repaints():
    editor = _Editor()
    window = _Window()
    presenter = LegacyFramePresenter(editor, window)
    presenter.tick()

    editor.request_repaint()
    editor.request_repaint()
    assert presenter.tick() is True
    assert presenter.tick() is False

    editor.worker_ready = True
    assert presenter.tick() is True
    assert presenter.presented_frames == 3


def test_legacy_presenter_repaints_resize_and_forced_smoke_frames():
    editor = _Editor()
    window = _Window()
    presenter = LegacyFramePresenter(editor, window)
    presenter.tick()

    window.size = (1024, 768)
    assert presenter.tick() is True
    assert editor.render_calls[-1] == (1024, 768)

    assert presenter.tick(force=True) is True
    assert presenter.presented_frames == 3


def test_legacy_status_projection_requests_only_changed_frames():
    class _Status:
        text = "Ready"

    class _View:
        def __init__(self):
            self._statusbar = _Status()
            self.repaint_calls = 0

        def request_repaint(self):
            self.repaint_calls += 1

    view = _View()
    presentation = _LegacyPresentation(view)

    presentation.set_status("Worker complete")
    presentation.set_status("Worker complete")

    assert view._statusbar.text == "Worker complete"
    assert view.repaint_calls == 1
