from dataclasses import replace
from pathlib import Path
import json
import pytest
from diffusion_editor.multiview_studio.model import MultiviewProject
from diffusion_editor.multiview_studio.controller import MultiviewStudioController
from diffusion_editor.multiview_studio.stablegen_model import StableGenSettings
def test_pass_history_accept_undo_redo_branch_and_reload(tmp_path):
    paths=[tmp_path/f'{n}.glb' for n in range(4)]
    for p in paths:p.write_bytes(b'geometry')
    geometry=str(paths[0])
    c=MultiviewStudioController(replace(MultiviewProject(),geometry_path=geometry,shape_path=geometry))
    c.accept_stablegen(paths[1]);c.accept_stablegen(paths[2])
    assert c.project.geometry_path==geometry
    c.step_stablegen_history(-1);assert c.project.shape_path==str(paths[1])
    c.accept_stablegen(paths[3])
    assert c.project.stablegen_history==(geometry,str(paths[1]),str(paths[3]))
    c.set_stablegen_setting('prompt','blue vest');c.set_stablegen_setting('reference',str(tmp_path/'ref.png'))
    saved=c.save(tmp_path/'project.json')
    p=MultiviewProject.load(saved)
    assert p.stablegen==c.project.stablegen
    assert p.stablegen_history==c.project.stablegen_history
    c=MultiviewStudioController(p)
    c.step_stablegen_history(-1);c.step_stablegen_history(1)
    assert c.project.shape_path==str(paths[3])
    c.set_textured_shape_path(paths[2])
    with pytest.raises(ValueError,match='outside'):c.step_stablegen_history(-1)


def test_legacy_defaults_and_invalid_settings(tmp_path):
    path=MultiviewProject().save(tmp_path/'old.json')
    data=json.loads(path.read_text())
    for k in ['stablegen','stablegen_history','stablegen_history_index']:data.pop(k)
    path.write_text(json.dumps(data))
    p=MultiviewProject.load(path)
    assert p.stablegen==StableGenSettings() and not p.stablegen_history
    for kwargs in [dict(denoise=0),dict(ip_strength=float('nan')),dict(size=111),dict(seed=-1)]:
        with pytest.raises(ValueError):StableGenSettings(**kwargs)


def test_unsaved_pass_history_is_adopted_with_all_artifacts(tmp_path):
    from types import SimpleNamespace
    from diffusion_editor.multiview_studio.native_app import NativeMultiviewStudioApplication
    app=object.__new__(NativeMultiviewStudioApplication)
    session=tmp_path/'session';destination=tmp_path/'saved';destination.mkdir()
    geometry=session/'shape-runs'/'source'/'model.glb';geometry.parent.mkdir(parents=True);geometry.write_bytes(b'geometry')
    material=session/'texture-runs'/'stablegen-test'/'candidate.glb';material.parent.mkdir(parents=True);material.write_bytes(b'textured')
    (material.parent/'request.json').write_text('{}')
    app.controller=MultiviewStudioController(replace(MultiviewProject(),geometry_path=str(geometry),shape_path=str(geometry)))
    app.controller.accept_stablegen(material)
    app._unsaved_workspace=SimpleNamespace(name=str(session))
    app._adopt_unsaved_artifacts(destination/'project.json')
    p=app.controller.project
    assert all(Path(path).is_relative_to(destination) for path in p.stablegen_history)
    assert Path(p.shape_path).parent.joinpath('request.json').is_file()
    assert p.stablegen_history[p.stablegen_history_index]==p.shape_path
    app.controller.step_stablegen_history(-1)
    assert app.controller.project.shape_path==app.controller.project.geometry_path


def test_stale_candidate_cannot_be_accepted(tmp_path):
    from types import SimpleNamespace
    from diffusion_editor.multiview_studio.stablegen_session import StableGenSession
    from diffusion_editor.multiview_studio.stablegen_service import fingerprint
    path=tmp_path/'source.glb';path.write_bytes(b'first')
    session=object.__new__(StableGenSession)
    session.app=SimpleNamespace(controller=MultiviewStudioController(replace(MultiviewProject(),geometry_path=str(path),shape_path=str(path))))
    session.snapshot=(str(path),str(path));session.source_hash=fingerprint(path);session.candidate=tmp_path/'candidate.glb'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='No current candidate'):session.accept()
    assert not session.app.controller.project.stablegen_history


def test_gpu_cancel_terminates_and_reaps_worker(tmp_path, monkeypatch):
    import threading
    from diffusion_editor.multiview_studio import stablegen_service as module
    calls=[]
    class Process:
        returncode=None
        def poll(self): return None
        def terminate(self): calls.append('terminate')
        def wait(self, timeout=None): calls.append('wait');self.returncode=-15
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:Process())
    class Cancel:
        def is_set(self): return False
        def wait(self, timeout): return True
    cancel=Cancel()
    service=module.StableGenService()
    with pytest.raises(RuntimeError,match='cancelled'):
        service.gpu(tmp_path,'prepare',cancel)
    assert calls==['terminate','wait']
    assert service.process is None


def test_gpu_failure_reports_diagnostics_and_clears_process(tmp_path, monkeypatch):
    import threading
    from diffusion_editor.multiview_studio import stablegen_service as module
    class Process:
        returncode=1
        def poll(self): return self.returncode
    def launch(*args,**kwargs):
        kwargs['stdout'].write('Invalid UV atlas');kwargs['stdout'].flush()
        return Process()
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    service=module.StableGenService()
    with pytest.raises(RuntimeError,match='Invalid UV atlas'):
        service.gpu(tmp_path,'prepare',threading.Event())
    assert service.process is None


def test_legacy_comfy_url_is_removed_on_save(tmp_path):
    path=MultiviewProject().save(tmp_path/'old.json')
    data=json.loads(path.read_text())
    data['stablegen']['comfy_url']='http://127.0.0.1:8188'
    path.write_text(json.dumps(data))
    loaded=MultiviewProject.load(path)
    loaded.save(path)
    assert 'comfy_url' not in json.loads(path.read_text())['stablegen']
    assert loaded.stablegen.ip_adapter.endswith('.safetensors')


def test_generation_returns_image_without_projection_and_removes_stale_candidate(tmp_path, monkeypatch):
    import threading
    from PIL import Image
    from diffusion_editor.multiview_studio.stablegen_service import StableGenService
    reference=tmp_path/'ref.png';Image.new('RGB',(16,16)).save(reference)
    (tmp_path/'request.json').write_text('{}')
    (tmp_path/'candidate.png').write_bytes(b'stale')
    (tmp_path/'candidate.glb').write_bytes(b'stale')
    service=StableGenService()
    operations=[]
    def gpu(root, operation, cancel):
        operations.append(operation)
        if operation=='generate':
            assert not (root/'candidate.png').exists()
            assert not (root/'candidate.glb').exists()
            Image.new('RGB',(16,16)).save(root/'candidate.png')
        else:
            pytest.fail('Image generation must not project')
    monkeypatch.setattr(service,'gpu',gpu)
    result=service.generate(tmp_path,StableGenSettings(reference=str(reference)),threading.Event())
    assert result==tmp_path/'candidate.png'
    assert operations==['generate']
    assert not (tmp_path/'candidate.glb').exists()


def test_failed_image_generation_never_projects_old_candidate(tmp_path, monkeypatch):
    import threading
    from PIL import Image
    from diffusion_editor.multiview_studio.stablegen_service import StableGenService
    reference=tmp_path/'ref.png';Image.new('RGB',(16,16)).save(reference)
    (tmp_path/'request.json').write_text('{}')
    (tmp_path/'candidate.glb').write_bytes(b'stale')
    service=StableGenService()
    def gpu(root, operation, cancel):
        assert operation=='generate'
        raise RuntimeError('model load failed')
    monkeypatch.setattr(service,'gpu',gpu)
    with pytest.raises(RuntimeError,match='model load failed'):
        service.generate(tmp_path,StableGenSettings(reference=str(reference)),threading.Event())
    assert not (tmp_path/'candidate.glb').exists()


def test_pre_cancelled_job_does_not_start_process(tmp_path, monkeypatch):
    import threading
    from diffusion_editor.multiview_studio import stablegen_service as module
    def launch(*args,**kwargs): pytest.fail('Cancelled job launched a worker')
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    cancel=threading.Event();cancel.set()
    with pytest.raises(RuntimeError,match='cancelled'):
        module.StableGenService().gpu(tmp_path,'generate',cancel)


def test_native_stablegen_fractional_controls_preserve_values():
    from types import SimpleNamespace
    from termin.gui_native import tc_ui_document_create, tc_ui_document_destroy
    from diffusion_editor.multiview_studio.native_view import NativeMultiviewStudioView
    from diffusion_editor.multiview_studio.stablegen_session import StableGenSession
    class Actions:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    document=tc_ui_document_create()
    view=NativeMultiviewStudioView(document,Actions(),request_repaint=lambda:None,texture_lease_factory=lambda:None)
    controller=MultiviewStudioController()
    app=SimpleNamespace(document=document,view=view,controller=controller,_safe=lambda f:f())
    session=StableGenSession(app)
    try:
        for field, value in [('denoise',.5),('cfg',1.25),('control_strength',.75),('ip_strength',.65)]:
            control=session.controls[field]
            assert control.decimals>0
            control.value=value
            assert control.value==pytest.approx(value)
            assert getattr(controller.project.stablegen,field)==pytest.approx(value)
        assert session.controls['steps'].decimals==0
        assert session.controls['seed'].decimals==0
    finally:
        session.close()
        view.close()
        tc_ui_document_destroy(document)


def test_fractional_setting_after_integer_value_from_json():
    controller=MultiviewStudioController(replace(MultiviewProject(),stablegen=StableGenSettings(denoise=1)))
    controller.set_stablegen_setting('denoise',.5)
    assert controller.project.stablegen.denoise==.5


def test_import_rejects_resized_camera_image_without_losing_candidate(tmp_path):
    from PIL import Image
    from diffusion_editor.multiview_studio.stablegen_service import StableGenService
    Image.new('RGB',(32,24)).save(tmp_path/'input-rgb.png')
    Image.new('RGB',(32,24),'red').save(tmp_path/'candidate.png')
    before=(tmp_path/'candidate.png').read_bytes()
    Image.new('RGB',(64,48)).save(tmp_path/'wrong.png')
    with pytest.raises(ValueError,match='dimensions'):
        StableGenService().import_image(tmp_path,tmp_path/'wrong.png')
    assert (tmp_path/'candidate.png').read_bytes()==before


def test_projection_is_an_explicit_separate_worker(tmp_path, monkeypatch):
    import threading
    from PIL import Image
    from diffusion_editor.multiview_studio.stablegen_service import StableGenService
    for name in ('input-rgb.png','candidate.png'):
        Image.new('RGB',(32,24)).save(tmp_path/name)
    service=StableGenService();operations=[]
    def gpu(root,operation,cancel):
        operations.append(operation)
        (root/'candidate.glb').write_bytes(b'projected')
    monkeypatch.setattr(service,'gpu',gpu)
    assert service.project(tmp_path,threading.Event()).read_bytes()==b'projected'
    assert operations==['project']


def test_editor_return_keeps_document_and_patch(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from PIL import Image
    from diffusion_editor.document.layer_stack import LayerStack
    from diffusion_editor.multiview_studio.image_editor_bridge import return_image
    stack=LayerStack()
    pixels=np.full((24,32,4),255,np.uint8)
    stack.init_from_image(pixels)
    stack.active_layer.patch_rect=(2,3,20,21)
    Image.fromarray(pixels).save(tmp_path/'editor-input.png')
    events=[]
    app=SimpleNamespace(layer_stack=stack,mark_document_saved=lambda p:events.append(p),request_stop=lambda:events.append('stop'))
    return_image(app,tmp_path)
    restored=LayerStack();restored.load_project(str(tmp_path/'image-edit.deproj'))
    assert restored.active_layer.patch_rect==(2,3,20,21)
    assert (tmp_path/'editor-return.json').is_file()
    assert not (tmp_path/'candidate.glb').exists()
    assert events[-1]=='stop'


def test_editor_cannot_return_resized_canvas(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from PIL import Image
    from diffusion_editor.document.layer_stack import LayerStack
    from diffusion_editor.multiview_studio.image_editor_bridge import return_image
    stack=LayerStack();stack.init_from_image(np.full((48,64,4),255,np.uint8))
    Image.new('RGB',(32,24)).save(tmp_path/'editor-input.png')
    with pytest.raises(ValueError,match='dimensions'):
        return_image(SimpleNamespace(layer_stack=stack),tmp_path)
    assert not (tmp_path/'editor-return.json').exists()
