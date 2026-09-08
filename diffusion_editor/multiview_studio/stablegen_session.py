"""Native Studio UI and acceptance lifecycle for single-camera texture passes."""
from dataclasses import replace
import json
from pathlib import Path
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from termin.base import MouseButton
from termin.gui_native import Point, PointerEventType, Size, FileDialogMode
from .stablegen_service import StableGenService, fingerprint


class StableGenSession:
    def __init__(self, app):
        self.app=app;self.view=app.view;self.doc=app.document
        self.service=StableGenService();self.root=None;self.candidate=None;self.image_candidate=None;self.snapshot=None
        self.syncing=False;self.busy=False;self.drawing=False;self.erase=False;self.previous=None
        self.controls={};self.buttons={};self.connections=[]
        group=self.doc.create_group_box('Texture · camera image and projection')
        group.widget.stable_id='multiview-studio.stablegen'
        content=self.doc.create_vstack('StableGenPassSettings');content.set_layout_spacing(4.)
        self.group=group
        def button(name,label,callback):
            b=self.doc.create_button(label);b.widget.stable_id='stablegen.'+name
            self.connections.append(b.connect_clicked(lambda: app._safe(callback)))
            content.add_preferred_child(b.widget);self.buttons[name]=b
        button('capture','1. Edit current camera',self.capture)
        button('editor','2. Edit image · Patch / model selection',self.edit_image)
        button('import','Import edited image...',lambda:app._show_file_dialog(FileDialogMode.OpenFile,'Images | *.png *.jpg *.jpeg *.webp',self.import_image))
        content.add_preferred_child(self.doc.create_label('Optional quick recipe: SDXL + Depth + IPAdapter'))
        for field,label in [('prompt','Prompt'),('negative','Negative prompt'),('reference','IPAdapter reference path')]:
            content.add_preferred_child(self.doc.create_label(label))
            control=self.doc.create_text_input('');control.widget.stable_id='stablegen.'+field
            self.connections.append(control.connect_changed(lambda value,name=field:self.change(name,value)))
            self.controls[field]=control;content.add_preferred_child(control.widget)
        button('browse','Choose reference image...',lambda:app._show_file_dialog(FileDialogMode.OpenFile,'Images | *.png *.jpg *.jpeg *.webp',lambda path:self.change('reference',path)))
        button('reference','Use Front as reference',lambda:self.change('reference',app.controller.project.front_path))
        for field,label,value,low,high,step in [('seed','Seed',31415,0,2147483647,1),('steps','Steps',8,1,100,1),('denoise','Denoise',.55,.01,1,.05),('control_strength','Depth ControlNet',.75,0,2,.05),('ip_strength','IPAdapter',.65,0,2,.05),('cfg','CFG',1.5,0,20,.1)]:
            spin = self.view._float_spin if isinstance(value, float) else self.view._spin
            self.controls[field]=spin(content,label,'stablegen.'+field,value,low,high,step,lambda value,name=field:self.change(name,value))
        self.brush=self.view._spin(content,'Brush radius (pixels)','stablegen.brush',20,1,160,1,lambda v:None)
        self.softness=self.view._spin(content,'Mask softness (pixels)','stablegen.softness',3,0,32,1,lambda v:None)
        content.add_preferred_child(self.doc.create_label('Paint in the captured image. Left: add, right: erase.'))
        button('all','Select visible surface',lambda:self.set_mask(True))
        button('clear','Clear mask',lambda:self.set_mask(False))
        button('generate','Generate image · SDXL + Depth + IPAdapter',self.generate)
        button('project','3. Project image onto mesh',self.project_image)
        button('another','Another seed',self.another)
        button('before','Show before',self.show_before)
        button('after','Show candidate',self.show_after)
        button('apply','4. Apply projected texture',self.accept)
        button('discard','Discard / close pass',self.discard)
        button('undo','Undo texture pass',lambda:self.history(-1))
        button('redo','Redo texture pass',lambda:self.history(1))
        group.set_content(content);self.view.left_content.add_preferred_child(group.widget)
        self.canvas=self.view.texture_pass_canvas
        self.connections.append(self.canvas.connect_pointer_input(self.pointer))
        self.relay=self.doc.create_scene_view();self.relay.widget.min_size=Size(0,0);self.relay.widget.preferred_size=Size(0,0)
        self.relay.set_pointer_handler(self.captured_pointer)
        self.view.left_content.add_fixed_child(self.relay.widget,0)
        self.refresh()

    def change(self,field,value):
        if not self.syncing and not self.busy:
            self.app._safe(lambda:self.app.controller.set_stablegen_setting(field,value))

    def refresh(self):
        if self.root is not None and self.snapshot != self.current() and not self.busy:
            self.root=None;self.candidate=None;self.image_candidate=None;self.snapshot=None
            self.canvas.widget.visible=False;self.view.selected_image.widget.visible=True
        if self.root is not None:
            self.view.selected_title.text = 'StableGen · captured camera'
            self.view.selected_path.text = 'Fixed view: paint a mask; capture again to change camera'
        self.syncing=True
        try:
            settings=self.app.controller.project.stablegen
            for field,c in self.controls.items():
                value=getattr(settings,field)
                if isinstance(value,str): c.text=value
                else:c.value=float(value)
        finally:self.syncing=False
        self.set_busy(self.busy)

    def set_busy(self,busy):
        self.busy=busy
        for c in self.controls.values():c.widget.enabled=not busy
        for b in self.buttons.values():b.widget.enabled=not busy
        for name in ('all','clear','generate','another','discard','editor','import'):
            self.buttons[name].widget.enabled=not busy and self.root is not None
        for name in ('before','after','project'):
            self.buttons[name].widget.enabled=not busy and self.image_candidate is not None
        self.buttons['apply'].widget.enabled=not busy and self.candidate is not None
        p=self.app.controller.project;i=p.stablegen_history_index
        current=0<=i<len(p.stablegen_history) and p.stablegen_history[i]==p.shape_path
        self.buttons['undo'].widget.enabled=not busy and current and i>0
        self.buttons['redo'].widget.enabled=not busy and current and i+1<len(p.stablegen_history)

    def current(self):
        p=self.app.controller.project
        return (p.geometry_path,p.shape_path)

    def valid(self):
        return self.snapshot is not None and self.current()==self.snapshot and fingerprint(self.snapshot[1])==self.source_hash

    def show_model(self,path):
        self.app.reconstruction_viewport.load_glb(str(path),fit_camera=False)
        # Keep application bookkeeping aligned, so settings edits don't refit.
        self.app._loaded_shape_path=str(path)
        self.app._displayed_mesh_signature=('main',str(path))

    def submit(self,work,finish):
        if self.app._job_active(): raise ValueError('Another operation is running')
        self.app._cancel=threading.Event();self.app._set_busy(True)
        self.app._qwen.shutdown()
        future=self.app._executor.submit(work,self.app._cancel)
        self.app._active_future=future
        def done():
            self.app._active_future=None;self.app._set_busy(False)
            try:finish(future.result())
            except Exception as e:
                if 'cancel' in str(e).lower():self.app.view.set_status('StableGen pass cancelled')
                else:self.app._show_error('StableGen texture pass',str(e))
            self.refresh()
        future.add_done_callback(lambda f:self.app._post(done))

    def capture(self):
        if self.app._selected_mesh_index!=0: raise ValueError('Select the whole model before capturing a texture pass')
        p=self.app.controller.project
        if not p.shape_path or not Path(p.shape_path).is_file(): raise ValueError('Load or build a model first')
        self.discard()
        v=self.app.reconstruction_viewport
        width,height=v.surface.size
        size=p.stablegen.size
        if width>=height: w=size;h=max(64,round(size*height/width/64)*64)
        else:h=size;w=max(64,round(size*width/height/64)*64)
        # Preserve viewport framing exactly. Output dimensions may be rounded,
        # but the projection matrix keeps the original viewport aspect ratio.
        convert=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],float)
        transform=np.asarray(v._model_transform,dtype=float)@convert
        mvp=v._camera.mvp(width,height)@transform
        eye=np.linalg.inv(transform)@np.array([*tuple(v._camera._camera.eye),1.])
        camera=dict(mvp=mvp.tolist(),eye=eye[:3].tolist(),size=[w,h],viewport_size=[width,height],coordinates='gltf-to-vulkan-clip')
        self.snapshot=self.current();self.source_hash=fingerprint(p.shape_path)
        path=self.app.controller.project_path or self.app._unsaved_project_path
        self.app.view.set_status('Preparing captured RGB and depth...')
        def finish(root):
            if not self.valid(): raise ValueError('Captured model is stale; capture again')
            self.root=root;self.mask=Image.open(root/'mask.png').convert('L');self.rgb=Image.open(root/'input-rgb.png').convert('RGB')
            self.canvas.widget.visible=True;self.view.selected_image.widget.visible=False
            self.preview_mask();self.canvas.fit_in_view()
            self.app.view.set_status('Paint a mask in the captured view, then generate a candidate')
        self.submit(lambda cancel:self.service.prepare(p.shape_path,path,camera,p.stablegen,cancel),finish)

    def preview_mask(self, original=False):
        if self.root is None:return
        base=self.rgb
        if not original and self.image_candidate is not None:
            base=Image.open(self.image_candidate).convert('RGB')
        a=np.asarray(base).copy();m=np.asarray(self.mask)/255*.4
        a=np.rint(a*(1-m[:,:,None])+np.array([255,50,40])*m[:,:,None]).astype(np.uint8)
        Image.fromarray(a).save(self.root/'mask-preview.png')
        self.view._set_preview('stablegen:pass',self.canvas,str(self.root/'mask-preview.png'),max_size=(2048,2048))
        self.app.composition.request_repaint()

    def set_mask(self,all_visible):
        if self.root is None:return
        self.mask=Image.open(self.root/'visible-mask.png').convert('L') if all_visible else Image.new('L',self.rgb.size)
        if self.candidate is not None:
            self.show_model(self.snapshot[1]);self.candidate=None;self.refresh()
        self.preview_mask()

    def dab(self,point):
        p=(float(point.x),float(point.y));radius=float(self.brush.value)
        draw=ImageDraw.Draw(self.mask);color=0 if self.erase else 255
        if self.previous:draw.line([self.previous,p],fill=color,width=max(1,round(radius*2)))
        draw.ellipse((p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius),fill=color)
        self.previous=p
        if self.candidate is not None:
            self.show_model(self.snapshot[1]);self.candidate=None;self.refresh()
        self.preview_mask()

    def pointer(self,point,event):
        if self.busy or self.root is None or event.type!=PointerEventType.Down:return
        self.erase=int(event.button)==int(MouseButton.RIGHT)
        self.drawing=True;self.previous=None;self.dab(point)
        self.doc.set_pointer_capture(self.relay.handle)

    def captured_pointer(self,point,event):
        if not self.drawing:return False
        self.dab(self.canvas.widget_to_image(Point(event.x,event.y)))
        if event.type in (PointerEventType.Up,PointerEventType.Cancel):
            self.drawing=False;self.previous=None
            if self.doc.pointer_capture==self.relay.handle:self.doc.release_pointer_capture(self.relay.handle)
        return True

    def save_mask(self, required=True):
        mask=self.mask.filter(ImageFilter.GaussianBlur(float(self.softness.value)))
        visible=np.asarray(Image.open(self.root/'visible-mask.png').convert('L'))/255
        mask=Image.fromarray(np.rint(np.asarray(mask)*visible).astype(np.uint8))
        if required and not np.asarray(mask).any():
            raise ValueError('Paint a nonempty projection mask on the visible model')
        mask.save(self.root/'mask.png')

    def invalidate_projection(self):
        if self.candidate is not None:
            self.show_model(self.snapshot[1])
        self.candidate=None
        self.refresh()

    def image_ready(self, path):
        if path is None:
            self.app.view.set_status('Editor closed; image was not returned')
            return
        if not self.valid():raise ValueError('Image is stale; capture the current model again')
        self.invalidate_projection()
        self.image_candidate=path
        self.show_after()
        self.refresh()
        self.app.view.set_status('Image ready. Edit further or press Project image onto mesh.')

    def generate(self):
        if self.root is None or not self.valid():raise ValueError('Capture the current model again')
        settings=self.app.controller.project.stablegen
        if not settings.prompt.strip():raise ValueError('Enter a prompt')
        if not settings.reference:raise ValueError('Choose an IPAdapter reference (or Use Front as reference)')
        self.save_mask()
        self.invalidate_projection()
        self.image_candidate=None
        self.app.view.set_status('Generating an image with SDXL + Depth ControlNet + IPAdapter...')
        self.submit(lambda cancel:self.service.generate(self.root,settings,cancel),self.image_ready)

    def edit_image(self):
        if self.root is None or not self.valid():raise ValueError('Capture the current model again')
        self.save_mask(required=False)
        self.app.view.set_status('Editing camera image in the main editor. Return it there when ready.')
        self.submit(lambda cancel:self.service.edit_image(self.root,cancel),self.image_ready)

    def import_image(self, path):
        if self.root is None or not self.valid():raise ValueError('Capture the current model again')
        self.image_ready(self.service.import_image(self.root,Path(path)))

    def project_image(self):
        if self.image_candidate is None or not self.valid():raise ValueError('Prepare an image before projection')
        self.save_mask()
        self.invalidate_projection()
        root=self.root
        signature=(fingerprint(root/'candidate.png'),fingerprint(root/'mask.png'))
        self.app.view.set_status('Projecting the image onto visible mesh texels...')
        def finish(path):
            if not self.valid():raise ValueError('Projection is stale; capture again')
            if signature!=(fingerprint(root/'candidate.png'),fingerprint(root/'mask.png')):
                raise ValueError('Image or mask changed during projection; project again')
            self.projected_signature=signature
            self.candidate=path
            self.show_after()
            self.app.view.set_status('Projected preview. Apply to keep the texture, or edit further.')
        self.submit(lambda cancel:self.service.project(root,cancel),finish)

    def another(self):
        self.change('seed',(self.app.controller.project.stablegen.seed+1)%(2**31))
        self.generate()

    def show_before(self):
        if not self.valid():raise ValueError('Pass is stale')
        self.show_model(self.snapshot[1]);self.preview_mask(original=True)

    def show_after(self):
        if self.image_candidate is None:return
        if not self.valid():raise ValueError('Pass is stale')
        if self.candidate is not None:self.show_model(self.candidate)
        self.view._set_preview('stablegen:pass',self.canvas,str(self.root/'candidate.png'),max_size=(2048,2048))

    def accept(self):
        if self.candidate is None or not self.valid():raise ValueError('No current candidate to apply')
        signature=(fingerprint(self.root/'candidate.png'),fingerprint(self.root/'mask.png'))
        if signature != self.projected_signature:
            raise ValueError('Image or mask changed; project again before applying')
        root=self.root;path=self.candidate
        self.show_model(path)
        self.app.controller.accept_stablegen(path)
        (root/'accepted.json').write_text(json.dumps(dict(shape=path.name,status='accepted'),indent=2))
        self.root=None;self.candidate=None;self.image_candidate=None;self.snapshot=None
        self.canvas.widget.visible=False;self.view.selected_image.widget.visible=True
        self.view._apply_selected_view()
        if self.app.controller.project_path:
            self.app._save_current_project()
        self.app.view.set_status('Texture pass applied; geometry unchanged')
        self.refresh()

    def discard(self):
        if self.app._job_active():raise ValueError('Cancel the running pass first')
        if self.candidate is not None:
            self.show_model(self.app.controller.project.shape_path)
        self.root=None;self.candidate=None;self.image_candidate=None;self.snapshot=None
        self.canvas.widget.visible=False;self.view.selected_image.widget.visible=True
        self.view._apply_selected_view()
        self.refresh()

    def history(self,delta):
        if self.app._job_active():raise ValueError('Cancel the running pass first')
        self.discard()
        self.app.controller.step_stablegen_history(delta)
        if self.app.controller.project_path:self.app._save_current_project()

    def close(self):
        if self.doc.pointer_capture==self.relay.handle:self.doc.release_pointer_capture(self.relay.handle)
        self.relay.set_pointer_handler(None)
        self.connections.clear()
