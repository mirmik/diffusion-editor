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
