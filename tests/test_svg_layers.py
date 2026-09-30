from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.document_service import DocumentService
from diffusion_editor.document.history import HistoryManager
from diffusion_editor.document.commands import (
    AddLayerCommand, AttachLayerToolCommand, DrawRectCommand, FlattenLayersCommand,
    SetLayerOpacityCommand, SetLayerVisibilityCommand,
)
from diffusion_editor.document.svg_layer import SvgLayer, parse_svg
from diffusion_editor.app.layer_tree import LayerTreeCoordinator

SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="32" height="24" viewBox="0 0 32 24">
<defs><linearGradient id="gradient"><stop stop-color="#00f"/><stop offset="1" stop-color="#0ff"/></linearGradient></defs>
<g id="rooms"><rect id="room" x="2" y="3" width="10" height="8" fill="#f00"/></g>
<path id="route" d="M 16 16 L 28 16" stroke="url(#gradient)" stroke-width="2"/>
<text id="label" x="1" y="23" font-size="5">Room A</text>
</svg>'''


def make_document():
    stack = LayerStack(tile_size=8)
    image = np.zeros((48, 64, 4), dtype=np.uint8)
    image[:] = (255, 255, 255, 255)
    stack.init_from_image(image)
    history = HistoryManager(stack.load_state)
    document = DocumentService(stack, history, stack.load_state)
    return stack, document


def test_svg_composite_update_history_visibility_offset_and_opacity():
    stack, doc = make_document()
    background = stack.active_layer
    identity = doc.svg.add(SVG, 'Rooms')
    layer = doc.svg.layer(identity)
    assert layer.node_type == 'svg' and not layer.accepts_pixel_edits
    assert (layer.width, layer.height) == (32, 24)
    np.testing.assert_array_equal(stack.composite()[5, 5], [255, 0, 0, 255])
    doc.svg.update_element('room', {'fill': '#00ff00'}, layer_id=identity)
    np.testing.assert_array_equal(stack.composite()[5, 5], [0, 255, 0, 255])
    changed = layer.svg_source
    assert doc.undo() == 'Update SVG Layer'
    assert layer.svg_source == SVG
    np.testing.assert_array_equal(stack.composite()[5, 5], [255, 0, 0, 255])
    doc.redo()
    assert layer.svg_source == changed
    doc.svg.move(10, 5, identity)
    np.testing.assert_array_equal(stack.composite()[10, 15], [0, 255, 0, 255])
    np.testing.assert_array_equal(stack.composite()[5, 5], [255, 255, 255, 255])
    doc.execute(SetLayerOpacityCommand(layer, 0.5))
    pixel = stack.composite()[10, 15]
    assert 180 < pixel[0] < 190 and pixel[1] == 255
    doc.execute(SetLayerVisibilityCommand(layer, False))
    np.testing.assert_array_equal(stack.composite(), background.image)


def test_roundtrip_preserves_source_id_tree_and_regenerates_cache(tmp_path):
    stack, doc = make_document()
    identity = doc.svg.add(SVG)
    layer = doc.svg.layer(identity)
    doc.execute(AddLayerCommand('Notes'))
    child = stack.active_layer
    stack.move_layer(child, layer, 0)
    stack.active_layer = layer
    path = tmp_path / 'plan.deproj'
    stack.save_project(str(path))
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        svg_entry = next(x for x in manifest['layers'] if x['type'] == 'svg')
        assert archive.read(svg_entry['svg_file']).decode() == SVG
        assert manifest['format_version'] == 9
        # A corrupted preview must not replace the authoritative source.
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as altered:
            for name in archive.namelist():
                data = archive.read(name)
                if name == svg_entry['image_file']:
                    buf = io.BytesIO()
                    np.save(buf, np.zeros((24, 32, 4), dtype=np.uint8))
                    data = buf.getvalue()
                altered.writestr(name, data)
    restored = LayerStack()
    restored.load_state(output.getvalue())
    recovered = restored.find_layer_by_id(identity)
    assert isinstance(recovered, SvgLayer)
    assert recovered.svg_source == SVG
    assert recovered.children[0].id == child.id
    assert recovered.children[0].parent is recovered
    assert restored.active_layer is recovered
    np.testing.assert_array_equal(recovered.image[5, 5], [255, 0, 0, 255])
    assert not recovered.image.flags.writeable


def test_invalid_update_and_missing_element_are_atomic():
    stack, doc = make_document()
    identity = doc.svg.add(SVG)
    before = stack.composite().copy()
    rev = stack.revision
    for bad in ('<svg', '<html/>', SVG.replace('id="route"', 'id="room"'),
                SVG.replace('fill="#f00"', 'fill="url(file:///tmp/secret)"')):
        with pytest.raises(ValueError):
            doc.svg.update(bad, identity)
        assert doc.svg.source(identity) == SVG
        assert stack.revision == rev
        np.testing.assert_array_equal(stack.composite(), before)
    with pytest.raises(ValueError, match='not found'):
        doc.svg.update_element('missing', {'x': '10'})
    assert doc.undo() == 'Add SVG Layer'
    assert stack.find_layer_by_id(identity) is None
    assert doc.redo() == 'Add SVG Layer'


def test_rasterize_preserves_children_identity_solo_and_can_undo():
    stack, doc = make_document()
    identity = doc.svg.add(SVG)
    vector = doc.svg.layer(identity)
    doc.execute(AddLayerCommand('Notes'))
    child = stack.active_layer
    stack.move_layer(child, vector, 0)
    stack.active_layer = vector
    stack.set_solo_layer(vector)
    before = stack.composite().copy()
    doc.svg.rasterize(identity)
    raster = stack.find_layer_by_id(identity)
    assert type(raster) is Layer
    assert raster.accepts_pixel_edits and raster.image.flags.writeable
    assert raster.children == (child,) and child.parent is raster
    assert stack.solo_layer_id == identity and stack.active_layer is raster
    np.testing.assert_array_equal(stack.composite(), before)
    doc.execute(DrawRectCommand(raster, 0, 0, 5, 5))
    doc.undo()
    assert doc.undo() == 'Rasterize SVG Layer'
    assert stack.find_layer_by_id(identity) is vector
    assert vector.children == (child,) and child.parent is vector
    assert vector.svg_source == SVG
    doc.redo()
    assert stack.find_layer_by_id(identity) is raster


def test_pixel_commands_reject_svg_and_layer_tree_exposes_vector():
    stack, doc = make_document()
    doc.svg.add(SVG)
    vector = stack.active_layer
    tree = LayerTreeCoordinator(stack, doc)
    assert tree.state.roots[0].node_type == 'svg'
    assert not tree.state.can_attach_tool
    assert tree.state.can_flatten
    with pytest.raises(ValueError, match='Rasterize'):
        doc.execute(DrawRectCommand(vector, 0, 0, 5, 5))
    doc.execute(FlattenLayersCommand())
    assert type(stack.active_layer) is Layer
    doc.undo()
    assert isinstance(stack.active_layer, SvgLayer)
    tree.close()


def test_resize_updates_viewport_and_undo_without_modifying_source():
    stack, doc = make_document()
    doc.svg.add(SVG)
    doc.svg.update(SVG, width=64, height=48)
    vector = stack.active_layer
    assert (vector.width, vector.height) == (64, 48)
    np.testing.assert_array_equal(vector.image[10, 10], [255, 0, 0, 255])
    assert vector.svg_source == SVG
    doc.undo()
    assert (vector.width, vector.height) == (32, 24)


def test_export_import_and_edit_label(tmp_path):
    stack, doc = make_document()
    doc.svg.add(SVG)
    doc.svg.update_element('label', text='Терминал')
    path = tmp_path / 'layer.svg'
    doc.svg.export_file(str(path))
    assert 'Терминал' in path.read_text()
    new_id = doc.svg.import_file(str(path))
    assert doc.svg.source(new_id) == path.read_text()
    assert any(e['id'] == 'rooms' for e in doc.svg.elements(new_id))


@pytest.mark.parametrize('body', [
    '<image href="file:///tmp/image.png"/>', '<image href="https://example.com/a.png"/>',
    '<script/>', '<foreignObject/>', '<animate/>', '<style>@import "file.css";</style>',
])
def test_external_or_dynamic_svg_is_rejected(body):
    with pytest.raises(ValueError):
        parse_svg(f'<svg xmlns="http://www.w3.org/2000/svg">{body}</svg>')


def test_svg_history_resolves_ids_after_snapshot_restore():
    stack, doc = make_document()
    identity = doc.svg.add(SVG)
    doc.svg.update_element('room', {'fill': 'blue'})
    doc.execute(FlattenLayersCommand())
    doc.undo()
    doc.undo()
    assert doc.svg.source(identity) == SVG
    np.testing.assert_array_equal(stack.composite()[5, 5], [255, 0, 0, 255])
    doc.undo()
    assert stack.find_layer_by_id(identity) is None
    doc.redo()
    doc.redo()
    np.testing.assert_array_equal(stack.composite()[5, 5], [0, 0, 255, 255])
