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
    Image.new('RGB',(16,16),'red').save(tmp_path/'candidate.png')
    (tmp_path/'candidate.glb').write_bytes(b'stale')
    Image.new('RGB',(16,16)).save(tmp_path/'input-rgb.png')
    service=StableGenService()
    operations=[]
    def gpu(root, operation, cancel):
        operations.append(operation)
        if operation=='generate':
            assert (root/'generation-input.png').read_bytes()==(root/'input-rgb.png').read_bytes()
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
    service.generate(tmp_path,StableGenSettings(reference=str(reference)),threading.Event())
    assert operations==['generate','generate']


def test_failed_image_generation_never_projects_old_candidate(tmp_path, monkeypatch):
    import threading
    from PIL import Image
    from diffusion_editor.multiview_studio.stablegen_service import StableGenService
    reference=tmp_path/'ref.png';Image.new('RGB',(16,16)).save(reference)
    (tmp_path/'request.json').write_text('{}')
    (tmp_path/'candidate.glb').write_bytes(b'stale')
    Image.new('RGB',(16,16)).save(tmp_path/'input-rgb.png')
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




def test_patch_crops_controls_together_and_preserves_outside():
    import numpy as np
    from PIL import Image
    from diffusion_editor.multiview_studio.stablegen_patch import prepare_patch,paste_patch
    source=Image.new('RGB',(100,80),(10,20,30))
    mask=Image.new('L',source.size)
    from PIL import ImageDraw
    ImageDraw.Draw(mask).rectangle((35,25,55,45),fill=255)
    depth=Image.new('RGB',source.size,(120,120,120))
    rgb,m,d,rect=prepare_patch(source,mask,depth,(30,20,70,60),512)
    assert rect==(30,20,70,60)
    assert rgb.size==m.size==d.size==(512,512)
    result=paste_patch(source,Image.new('RGB',rgb.size,'red'),mask,rect)
    a=np.asarray(result);before=np.asarray(source)
    changed=np.any(a!=before,axis=-1)
    assert changed.any()
    assert not changed[np.asarray(mask)==0].any()
    assert not changed[:20].any() and not changed[:,70:].any()
    assert result.size==source.size
    with pytest.raises(ValueError,match='intersect'):
        prepare_patch(source,mask,depth,(0,0,10,10),512)


def test_checkpoint_header_filter(tmp_path):
    import struct
    from diffusion_editor.multiview_studio.stablegen_checkpoints import is_sdxl_checkpoint
    path=tmp_path/'model.safetensors'
    def write(shape):
        header=json.dumps({'model.diffusion_model.input_blocks.0.0.weight':{'shape':shape},
                           'conditioner.embedders.1.model.text_projection':{'shape':[1280,1280]}}).encode()
        path.write_bytes(struct.pack('<Q',len(header))+header)
    write([320,4,3,3]);assert is_sdxl_checkpoint(path)
    write([320,9,3,3]);assert not is_sdxl_checkpoint(path)
    path.write_bytes(b'broken');assert not is_sdxl_checkpoint(path)


def test_model_and_optional_lora_survive_save(tmp_path):
    c=MultiviewStudioController()
    c.set_stablegen_setting('checkpoint','dreamshaperXL_lightningDPMSDE.safetensors')
    c.set_stablegen_setting('lora','')
    c.set_stablegen_setting('size',1024)
    loaded=MultiviewProject.load(c.save(tmp_path/'project.json'))
    assert loaded.stablegen.checkpoint=='dreamshaperXL_lightningDPMSDE.safetensors'
    assert loaded.stablegen.lora==''
    assert loaded.stablegen.size==1024


def test_mask_events_coalesce_without_png_io(tmp_path,monkeypatch):
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from diffusion_editor.multiview_studio.stablegen_session import StableGenSession
    from termin.gui_native import Point
    session=object.__new__(StableGenSession)
    session.root=tmp_path;session.rgb=Image.new('RGB',(256,256))
    session.mask=Image.new('L',(256,256));session.image_candidate=None;session.candidate=None
    session.previous=None;session.erase=False;session.brush=SimpleNamespace(value=8)
    session._preview_dirty=False;session._preview_options=(False,True)
    session.app=SimpleNamespace(composition=SimpleNamespace(request_repaint=lambda:None))
    uploads=[]
    lease=SimpleNamespace(texture=object(),set_rgba8=lambda pixels,encoding:uploads.append(pixels.copy()))
    session.view=SimpleNamespace(_leases={},_preview_sizes={},_texture_lease_factory=lambda:lease)
    session.canvas=SimpleNamespace(set_texture=lambda *args:None)
    def forbidden(*args,**kwargs):pytest.fail('PNG IO in input/render path')
    monkeypatch.setattr(Image.Image,'save',forbidden)
    monkeypatch.setattr(Image,'open',forbidden)
    for i in range(100):session.dab(Point(20+i,40))
    assert not uploads
    assert session.flush_preview()
    assert len(uploads)==1
    assert not session.flush_preview()
    assert np.asarray(session.mask)[40,80]==255
    assert uploads[0][40,80,0]>0


def test_sampling_mode_matches_editor_and_allows_override(tmp_path):
    from diffusion_editor.sdxl_sampling import resolve_prediction_type
    assert resolve_prediction_type('smoothYaoiBoys_v30Vpred.safetensors')=='v_prediction'
    assert resolve_prediction_type('RealVisXL.safetensors')=='epsilon'
    assert resolve_prediction_type('vpred.safetensors','epsilon')=='epsilon'
    assert resolve_prediction_type('unknown.safetensors','v_prediction')=='v_prediction'
    c=MultiviewStudioController()
    c.set_stablegen_setting('prediction_type','v_prediction')
    c.set_stablegen_setting('sampler','dpmpp_sde_karras')
    loaded=MultiviewProject.load(c.save(tmp_path/'project.json'))
    assert loaded.stablegen.prediction_type=='v_prediction'
    assert loaded.stablegen.sampler=='dpmpp_sde_karras'
    with pytest.raises(ValueError):
        StableGenSettings(prediction_type='invalid')
