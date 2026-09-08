from dataclasses import replace
from pathlib import Path
import json
import sys
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from diffusion_editor.multiview_studio.controller import MultiviewStudioController
from diffusion_editor.multiview_studio.model import MultiviewProject, Pixal3DSettings, ViewKey, all_view_keys
from diffusion_editor.multiview_studio.pixal3d_views import orbit_camera, prepare_views
from diffusion_editor.multiview_studio.pixal3d_generation import Pixal3DGenerator
from diffusion_editor.multiview_studio.native_app import NativeMultiviewStudioApplication


def test_roundtrip_settings_and_legacy_default(tmp_path):
    p = tmp_path / 'project.mvstudio.json'
    c = MultiviewStudioController()
    c.set_shape_backend('pixal3d')
    c.set_pixal3d_setting('steps', 24)
    c.set_pixal3d_setting('resolution', 1536)
    c.set_pixal3d_setting('fov', 35.5)
    c.set_pixal3d_setting('normalize_views', False)
    c.save(p)
    assert MultiviewProject.load(p) == c.project
    data = json.loads(p.read_text())
    del data['shape_backend']; del data['pixal3d']
    p.write_text(json.dumps(data))
    assert MultiviewProject.load(p).shape_backend == 'trellis'
    with pytest.raises(ValueError): c.set_shape_backend('unknown')
    with pytest.raises(ValueError): c.set_pixal3d_setting('resolution', 1280)
    with pytest.raises(ValueError): c.set_pixal3d_setting('fov', float('nan'))


def test_pixal_views_do_not_need_trellis_warmup_budget(tmp_path):
    p = replace(MultiviewProject(), shape_backend='pixal3d')
    assert p.validate_shape_request()
    for key in all_view_keys(): p = p.with_slot(key, image_path='image.png')
    p = replace(p, trellis=replace(p.trellis, total_steps=1, warmup_steps=1))
    assert not p.validate_shape_request()
    assert replace(p, shape_backend='trellis').validate_shape_request()


def test_orbit_cameras_are_right_handed_and_look_at_origin():
    for key in all_view_keys():
        m = np.asarray(orbit_camera(key, 3.1))
        assert np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3))
        assert np.isclose(np.linalg.det(m[:3, :3]), 1)
        assert np.allclose(m[:3, 3] - m[:3, 2] * 3.1, 0)
    assert np.allclose(orbit_camera(ViewKey('eye', 0), 3), [[1,0,0,0],[0,0,-1,-3],[0,1,0,0],[0,0,0,1]])


def test_preparation_uses_alpha_orders_front_and_preserves_aspect(tmp_path):
    path = tmp_path/'masked.png'
    a = np.zeros((100, 60, 4), np.uint8)
    a[10:90, 20:40] = [255, 80, 20, 255]
    Image.fromarray(a).save(path)
    p = MultiviewProject()
    for key in (ViewKey('low',90),ViewKey('eye',0),ViewKey('elevated',270)):
        p = p.with_slot(key, image_path=str(path))
    def forbidden(*a, **kw): raise AssertionError('alpha input must not use segmenter')
    out = tmp_path/'prepared'
    prepare_views(p.slots,out,Pixal3DSettings(),threading.Event(),forbidden)
    meta=json.loads((out/'transforms.json').read_text())
    assert [f['name'] for f in meta['frames']]==['eye-000','low-090','elevated-270']
    with Image.open(out/'eye-000.png') as im:
        alpha=np.array(im.getchannel('A')); ys,xs=np.nonzero(alpha>128)
        assert im.size==(1024,1024)
        assert (xs.max()-xs.min())/(ys.max()-ys.min())==pytest.approx(0.25,abs=0.01)
    assert np.array_equal(np.array(Image.open(path)),a)


def test_preparation_segments_rgb_and_rejects_empty_mask(tmp_path):
    path=tmp_path/'rgb.png'; Image.new('RGB',(40,30),'red').save(path)
    p=MultiviewProject().with_source('front',str(path)); calls=[]
    def segment(a,cancel,**kw):
        calls.append(a.shape); return np.full(a.shape[:2],255,np.uint8)
    out=tmp_path/'views'
    prepare_views(p.slots,out,Pixal3DSettings(),threading.Event(),segment)
    assert calls==[(30,40,3)]
    with pytest.raises(ValueError,match='Empty foreground'):
        prepare_views(p.slots,tmp_path/'empty',Pixal3DSettings(),threading.Event(),lambda a,*args,**kw: np.zeros(a.shape[:2],np.uint8))
    cancel=threading.Event();cancel.set()
    with pytest.raises(RuntimeError,match='cancelled'):
        prepare_views(p.slots,tmp_path/'cancel',Pixal3DSettings(),cancel,segment)


def test_silent_worker_can_be_cancelled_and_failure_keeps_log(tmp_path):
    g=Pixal3DGenerator(root=tmp_path)
    cancel=threading.Event(); timer=threading.Timer(0.3,cancel.set());timer.start()
    start=time.monotonic()
    try:
        with pytest.raises(RuntimeError,match='cancelled'):
            g._run_worker([sys.executable,'-c','import time; time.sleep(30)'],tmp_path,cancel)
    finally: timer.cancel()
    assert time.monotonic()-start<5
    assert g._process is None
    with pytest.raises(RuntimeError,match='deliberate error'):
        g._run_worker([sys.executable,'-c','raise RuntimeError("deliberate error")'],tmp_path,threading.Event())
    assert 'deliberate error' in (tmp_path/'worker.log').read_text()


def test_worker_result_relative_path_and_missing_result(tmp_path):
    g=Pixal3DGenerator(root=tmp_path)
    with pytest.raises(RuntimeError,match='no result'):
        g._run_worker([sys.executable,'-c','pass'],tmp_path,threading.Event())
    (tmp_path/'model.glb').write_bytes(b'glb')
    (tmp_path/'result.json').write_text(json.dumps({'status':'success','shape':'model.glb'}))
    assert g._run_worker([sys.executable,'-c','pass'],tmp_path,threading.Event())==tmp_path/'model.glb'


def test_application_dispatches_pixal_and_shutdowns_qwen(tmp_path):
    app=object.__new__(NativeMultiviewStudioApplication)
    p=replace(MultiviewProject().with_source('front','front.png'),shape_backend='pixal3d')
    app.controller=MultiviewStudioController(p);app.controller.project_path=tmp_path/'project.mvstudio.json'
    app._job_active=lambda:False;app._set_busy=lambda _:None
    app.view=SimpleNamespace(apply_project=lambda *a:None,set_status=lambda *a:None)
    events=[]
    app._qwen=SimpleNamespace(shutdown=lambda:events.append('qwen stopped'))
    app._pixal3d=SimpleNamespace(generate=lambda *a:None)
    app._trellis=SimpleNamespace(generate=lambda *a:None)
    def submit(fn,*args):
        events.append(fn); return Future()
    app._executor=SimpleNamespace(submit=submit)
    app.build_shape()
    assert events==['qwen stopped',app._pixal3d.generate]


def test_result_does_not_replace_a_changed_project(tmp_path):
    app=object.__new__(NativeMultiviewStudioApplication)
    old=replace(MultiviewProject(),shape_backend='pixal3d')
    app.controller=MultiviewStudioController(replace(old,qwen_seed=7))
    messages=[];app._set_busy=lambda _:None
    app.view=SimpleNamespace(set_status=messages.append)
    f=Future();f.set_result(tmp_path/'model.glb')
    app._finish_shape_generation(f,old)
    assert not app.controller.project.shape_path
    assert 'Ignored Pixal3D' in messages[-1]


def test_geometry_worker_exports_without_texture_pipeline(tmp_path, monkeypatch):
    """Build must never call full run_mv or the PBR baking exporter."""
    from diffusion_editor.multiview_studio import pixal3d_runner as runner
    from contextlib import nullcontext
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail('Build model invoked a texture operation')

    pipeline = SimpleNamespace(run_mv=forbidden)
    upstream = SimpleNamespace(
        load_views=lambda path: {'view_names': ['front']},
        check_main_view=lambda views: None,
        init_pipeline=lambda *a, **kw: pipeline,
        o_voxel=SimpleNamespace(postprocess=SimpleNamespace(to_glb=forbidden)),
    )
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    monkeypatch.setattr(runner.importlib.util, 'spec_from_file_location', lambda *a: spec)
    monkeypatch.setattr(runner.importlib.util, 'module_from_spec', lambda spec: upstream)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        __version__='fake', no_grad=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda n: 'fake',
                             max_memory_allocated=lambda: 0)))
    monkeypatch.setitem(sys.modules, 'pixal3d.pipelines', SimpleNamespace(rembg=SimpleNamespace()))
    vertices = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: vertices.copy()))
    mesh = SimpleNamespace(vertices=tensor, faces=tensor,
        fill_holes=lambda: calls.append('fill'),
        simplify=lambda **kw: calls.append(('simplify', kw)))
    monkeypatch.setattr(runner, 'generate_geometry', lambda *a: ([mesh], 1024))

    def postprocess(cache, output, settings, resolution, **kwargs):
        with np.load(cache) as data:
            np.testing.assert_array_equal(data['vertices'], vertices)
        assert settings['input_coordinates'] == 'gltf_y_up'
        assert settings['exact_face_count']
        assert settings['decimation_target'] == 1234
        assert settings['remesh'] and settings['cleanup']
        calls.append('shared postprocess')
        target = output / 'shape-post-test.glb'
        target.write_bytes(b'geometry')
        return target, {'output_counts': {'faces': 1234}}

    monkeypatch.setitem(sys.modules, 'trellis_mesh_postprocess', SimpleNamespace(run_mesh_postprocess=postprocess))
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(dict(protocol=1, root=str(tmp_path), views_dir='views',
        model_path='model', settings={'decimation_target': 1234})))
    runner.run(request)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['status'] == 'success'
    assert result['textured'] is False
    assert calls == ['shared postprocess']
    assert result['postprocess']['output_counts']['faces'] == 1234


def test_native_texture_settings_persist_and_do_not_use_trellis_schedule(tmp_path):
    from diffusion_editor.multiview_studio.model import Pixal3DTextureSettings
    p = replace(MultiviewProject().with_source('front', 'front.png'),
                shape_backend='pixal3d', geometry_path='mesh.glb')
    c = MultiviewStudioController(p)
    c.set_pixal3d_texture_setting('steps', 1)
    c.set_pixal3d_texture_setting('seed', 77)
    path = c.save(tmp_path / 'project.json')
    loaded = MultiviewProject.load(path)
    assert loaded.pixal3d_texture == Pixal3DTextureSettings(steps=1, seed=77)
    assert not loaded.validate_texture_request()
    assert loaded.texture == p.texture
    payload = json.loads(path.read_text()); payload.pop('pixal3d_texture')
    path.write_text(json.dumps(payload))
    assert MultiviewProject.load(path).pixal3d_texture == Pixal3DTextureSettings()
    with pytest.raises(ValueError):
        c.set_pixal3d_texture_setting('steps', 0)


def test_application_dispatches_native_texture(tmp_path):
    app = object.__new__(NativeMultiviewStudioApplication)
    mesh = tmp_path / 'mesh.glb'; mesh.write_bytes(b'mesh')
    p = replace(MultiviewProject().with_source('front', 'front.png'),
                shape_backend='pixal3d', geometry_path=str(mesh))
    app.controller = MultiviewStudioController(p)
    app.controller.project_path = tmp_path / 'project.json'
    app._job_active = lambda: False
    app._set_busy = lambda _: None
    app.view = SimpleNamespace(apply_project=lambda *a: None, set_status=lambda *a: None)
    events = []
    app._qwen = SimpleNamespace(shutdown=lambda: events.append('qwen stopped'))
    app._pixal3d = SimpleNamespace(generate_texture=lambda *a: None)
    def submit(fn, *args):
        events.append(fn)
        return Future()
    app._executor = SimpleNamespace(submit=submit)
    app.texture_model()
    assert events == ['qwen stopped', app._pixal3d.generate_texture]


def test_texture_request_snapshots_mesh_and_uses_native_settings(tmp_path, monkeypatch):
    root = tmp_path / 'runtime'; root.mkdir()
    (root / 'inference_mv.py').touch(); (root / 'pipeline_mv.json').touch()
    encoder = root / 'encoder'
    for suffix in ('.json', '.safetensors'):
        Path(str(encoder) + suffix).touch()
    monkeypatch.setenv('DIFFUSION_EDITOR_PIXAL3D_SHAPE_ENCODER', str(encoder))
    front = tmp_path / 'front.png'
    Image.new('RGBA', (20, 30), (255, 0, 0, 128)).save(front)
    mesh = tmp_path / 'mesh.glb'; mesh.write_bytes(b'original geometry')
    p = replace(MultiviewProject().with_source('front', str(front)),
                shape_backend='pixal3d', geometry_path=str(mesh))
    shutdowns = []
    segmenter = SimpleNamespace(segment=lambda *a: pytest.fail('alpha already present'),
                                shutdown=lambda: shutdowns.append(True))
    g = Pixal3DGenerator(python=Path(sys.executable), root=root, model_path=root, segmenter=segmenter)
    def worker(command, output, cancel, progress):
        r = json.loads((output / 'request.json').read_text())
        assert r['operation'] == 'texture'
        assert r['texture_settings'] == dict(seed=43, steps=12, texture_size=2048)
        assert (output / r['input_mesh']).read_bytes() == b'original geometry'
        assert (output / r['views_dir'] / 'transforms.json').is_file()
        assert output.parent.name == 'texture-runs'
        assert shutdowns == [True]
        return output / 'model.glb'
    monkeypatch.setattr(g, '_run_worker', worker)
    result = g.generate_texture(p, tmp_path / 'project.json', threading.Event())
    assert result.name == 'model.glb'
