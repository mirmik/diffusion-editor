from diffusion_editor.canvas.canvas_edit_session import CanvasEditSession


def test_edit_session_begin_records_callback_metadata():
    layer = object()
    session = CanvasEditSession()

    session.begin(label="Paint", target="image", layer=layer, pos=(3, 4))

    assert session.active is True
    assert session.label == "Paint"
    assert session.target == "image"
    assert session.layer is layer
    assert session.last_pos == (3, 4)
    assert session.dirty_rect is None


def test_edit_session_accumulates_dirty_rects():
    session = CanvasEditSession()
    session.begin(label="Paint", target="image", layer=None, pos=(0, 0))

    session.add_dirty((2, 3, 4, 5))
    session.add_dirty((1, 4, 6, 7))

    assert session.dirty_rect == (1, 3, 6, 7)


def test_edit_session_clear_resets_state():
    session = CanvasEditSession()
    session.begin(label="Paint", target="image", layer=None, pos=(0, 0))
    session.add_dirty((0, 0, 1, 1))
    session.move_to((2, 3))

    session.clear()

    assert session.active is False
    assert session.label is None
    assert session.target is None
    assert session.layer is None
    assert session.last_pos is None
    assert session.dirty_rect is None
