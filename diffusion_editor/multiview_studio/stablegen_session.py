"""Native Studio UI and acceptance lifecycle for single-camera texture passes."""
from dataclasses import replace
import json
from pathlib import Path
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from termin.base import MouseButton
from termin.gui_native import Point, PointerEventType, Size, FileDialogMode, Rect, SrgbColor
from termin.graphics import TextureEncoding
from .stablegen_service import StableGenService, fingerprint
from .stablegen_patch import patch_rect
from .stablegen_checkpoints import available_checkpoints, resolve_checkpoint, is_sdxl_checkpoint


class StableGenSession:
    def __init__(self, app):
        self.app=app;self.view=app.view;self.doc=app.document
        self.service=StableGenService();self.root=None;self.candidate=None;self.image_candidate=None;self.snapshot=None
        self.syncing=False;self.busy=False;self.drawing=False;self.erase=False;self.previous=None
        self._preview_dirty=False;self._preview_options=(False,True);self._cached_image=None;self._cached_image_path=None
        self.patch=None;self.patch_drag=None;self.patch_mode=False
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
        button('import','Import edited image...',lambda:app._show_file_dialog(FileDialogMode.OpenFile,'Images | *.png *.jpg *.jpeg *.webp',self.import_image))
        content.add_preferred_child(self.doc.create_label('SDXL checkpoint · ControlNet and IPAdapter stay enabled'))
        self.checkpoint_choices=available_checkpoints()
        self.checkpoint_combo=self.doc.create_combo_box()
        self.checkpoint_combo.widget.stable_id='stablegen.checkpoint'
        for name in self.checkpoint_choices:self.checkpoint_combo.add_item(name)
        self.connections.append(self.checkpoint_combo.connect_changed(
            lambda index,*_:self.choose_checkpoint(index)))
        content.add_preferred_child(self.checkpoint_combo.widget)
        button('checkpoint_file','Choose SDXL checkpoint...',lambda:app._show_file_dialog(
            FileDialogMode.OpenFile,'SDXL checkpoint | *.safetensors',self.select_checkpoint))
        self.lora_combo=self.doc.create_combo_box()
        self.lora_combo.widget.stable_id='stablegen.lora'
        self.lora_choices=['','sdxl_lightning_8step_lora.safetensors']
        for label in ('No acceleration LoRA','Lightning 8-step LoRA'):self.lora_combo.add_item(label)
        self.connections.append(self.lora_combo.connect_changed(
            lambda index,*_:self.change('lora',self.lora_choices[index]) if 0<=index<len(self.lora_choices) else None))
        content.add_preferred_child(self.lora_combo.widget)
        self.sampling_combos={}
        for field,labels,values in [
            ('prediction_type',('Prediction: Auto','Prediction: epsilon','Prediction: v_prediction'),('auto','epsilon','v_prediction')),
            ('sampler',('Sampler: Auto','Euler trailing','DPM++ SDE Karras'),('auto','euler','dpmpp_sde_karras'))]:
            combo=self.doc.create_combo_box()
            for label in labels:combo.add_item(label)
            self.connections.append(combo.connect_changed(
                lambda index,*_,name=field,options=values:self.change(name,options[index]) if 0<=index<len(options) else None))
            self.sampling_combos[field]=(combo,values)
            content.add_preferred_child(combo.widget)
        button('model_defaults','Use model sampling defaults',self.model_defaults)
        self.size_combo=self.doc.create_combo_box()
        for label in ('512','768','1024'):self.size_combo.add_item(label)
        self.connections.append(self.size_combo.connect_changed(
            lambda index,*_:self.change('size',(512,768,1024)[index]) if 0<=index<3 else None))
        content.add_preferred_child(self.doc.create_label('Patch working resolution (long side)'))
        content.add_preferred_child(self.size_combo.widget)
        button('patch','Draw Patch',self.toggle_patch)
        button('full_patch','Full image Patch',lambda:self.set_patch(None))
        self.patch_label=self.doc.create_label('Patch: full image')
        content.add_preferred_child(self.patch_label)
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
        button('generate','2. Generate image in Patch',self.generate)
        button('project','3. Project image onto mesh',self.project_image)
        button('another','Another seed',self.another)
        button('reset_image','Reset image to captured view',lambda:self.import_image(self.root/'input-rgb.png'))
        button('before','Show before',self.show_before)
        button('after','Show candidate',self.show_after)
        button('apply','4. Apply projected texture',self.accept)
        button('discard','Discard / close pass',self.discard)
        button('undo','Undo texture pass',lambda:self.history(-1))
        button('redo','Redo texture pass',lambda:self.history(1))
        group.set_content(content);self.view.left_content.add_preferred_child(group.widget)
        self.canvas=self.view.texture_pass_canvas
        self.canvas.set_paint_callback(self.paint_patch)
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
            for field,(combo,values) in self.sampling_combos.items():combo.selected_index=values.index(getattr(settings,field))
            if settings.checkpoint not in self.checkpoint_choices:
                self.checkpoint_choices.append(settings.checkpoint);self.checkpoint_combo.add_item(settings.checkpoint)
            self.checkpoint_combo.selected_index=self.checkpoint_choices.index(settings.checkpoint)
            if settings.lora not in self.lora_choices:
                self.lora_choices.append(settings.lora);self.lora_combo.add_item(settings.lora)
            self.lora_combo.selected_index=self.lora_choices.index(settings.lora)
            self.size_combo.selected_index=(512,768,1024).index(settings.size)
            for field,c in self.controls.items():
                value=getattr(settings,field)
                if isinstance(value,str): c.text=value
                else:c.value=float(value)
        finally:self.syncing=False
        self.set_busy(self.busy)

    def set_busy(self,busy):
        self.busy=busy
        for combo,_ in self.sampling_combos.values():combo.widget.enabled=not busy
        for combo in (self.checkpoint_combo,self.lora_combo,self.size_combo):combo.widget.enabled=not busy
        for c in self.controls.values():c.widget.enabled=not busy
        for b in self.buttons.values():b.widget.enabled=not busy
        for name in ('all','clear','generate','another','discard','import','reset_image','patch','full_patch'):
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
        self.patch=None;self.patch_drag=None;self.patch_mode=False
        self.buttons["patch"].set_text("Draw Patch");self.patch_label.text="Patch: full image"
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
            self.preview_mask();self.flush_preview();self.canvas.fit_in_view()
            self.app.view.set_status('Paint a mask in the captured view, then generate a candidate')
        self.submit(lambda cancel:self.service.prepare(p.shape_path,path,camera,p.stablegen,cancel),finish)

    def preview_mask(self, original=False, show_mask=True):
        if self.root is None:return
        self._preview_options=(original,show_mask)
        self._preview_dirty=True
        self.app.composition.request_repaint()

    def flush_preview(self):
        """Upload at most once per UI frame, without encoding or disk roundtrips."""
        if not self._preview_dirty:return False
        self._preview_dirty=False
        if self.root is None:return False
        original,show_mask=self._preview_options
        base=self.rgb
        if not original and self.image_candidate is not None:
            if self._cached_image_path != self.image_candidate:
                with Image.open(self.image_candidate) as image:
                    self._cached_image=image.convert('RGB')
                self._cached_image_path=self.image_candidate
            base=self._cached_image
        if show_mask:
            alpha=self.mask.point([round(v*.4) for v in range(256)])
            preview=Image.composite(Image.new('RGB',base.size,(255,50,40)),base,alpha)
        else:
            preview=base
        pixels=np.array(preview.convert('RGBA'),dtype=np.uint8,copy=True,order='C')
        key='stablegen:pass'
        lease=self.view._leases.get(key)
        if lease is None:
            lease=self.view._texture_lease_factory()
            self.view._leases[key]=lease
        lease.set_rgba8(pixels,TextureEncoding.SRGB)
        size=Size(base.width,base.height)
        self.view._preview_sizes[key]=size
        self.canvas.set_texture(lease.texture,size)
        return True

    def paint_patch(self,context):
        bounds=self.patch_drag or self.patch
        if self.root is None or bounds is None:return
        a=self.canvas.image_to_widget(Point(bounds[0],bounds[1]))
        b=self.canvas.image_to_widget(Point(bounds[2],bounds[3]))
        rect=Rect(min(a.x,b.x),min(a.y,b.y),abs(b.x-a.x),abs(b.y-a.y))
        context.stroke_rect(rect,SrgbColor(.08,.12,.08,.95),4.)
        context.stroke_rect(rect,SrgbColor(.25,1.,.38,.95),2.)

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

    def choose_checkpoint(self,index):
        if not self.syncing and not self.busy and 0<=index<len(self.checkpoint_choices):
            self.app._safe(lambda:self.select_checkpoint(self.checkpoint_choices[index]))

    def select_checkpoint(self,name):
        path=resolve_checkpoint(name)
        if not is_sdxl_checkpoint(path):
            raise ValueError('Choose a standard SDXL checkpoint compatible with this Depth ControlNet/IPAdapter')
        self.change('checkpoint',str(name))
        # Other checkpoints may already be distilled. Never stack Lightning implicitly.
        self.change('lora','')
        self.model_defaults()

    def model_defaults(self):
        settings=self.app.controller.project.stablegen
        accelerated=bool(settings.lora) or 'lightning' in settings.checkpoint.lower()
        for name,value in [('prediction_type','auto'),('sampler','auto'),
                           ('steps',8 if accelerated else 30),('cfg',1.5 if accelerated else 5.)]:
            self.change(name,value)

    def toggle_patch(self):
        self.patch_mode=not self.patch_mode
        self.buttons['patch'].set_text('Paint mask' if self.patch_mode else 'Draw Patch')
        self.app.view.set_status('Drag a Patch rectangle in the captured image' if self.patch_mode else 'Paint the mask inside the Patch')

    def set_patch(self,bounds):
        if self.root is None:return
        self.patch=None if bounds is None else patch_rect(bounds,self.rgb.size)
        self.patch_drag=None
        self.patch_label.text='Patch: full image' if self.patch is None else f'Patch: {self.patch}'
        (self.root/'patch.json').write_text(json.dumps(self.patch))
        self.app.composition.request_repaint()

    def pointer(self,point,event):
        if self.busy or self.root is None or event.type!=PointerEventType.Down:return
        if self.patch_mode:
            if int(event.button)!=int(MouseButton.LEFT):return
            self.patch_drag=(point.x,point.y,point.x,point.y)
            self.doc.set_pointer_capture(self.relay.handle)
            return
        self.erase=int(event.button)==int(MouseButton.RIGHT)
        self.drawing=True;self.previous=None;self.dab(point)
        self.doc.set_pointer_capture(self.relay.handle)

    def captured_pointer(self,point,event):
        if self.patch_drag is not None:
            p=self.canvas.widget_to_image(Point(event.x,event.y))
            x0,y0,_,_=self.patch_drag
            self.patch_drag=(x0,y0,p.x,p.y)
            if event.type in (PointerEventType.Up,PointerEventType.Cancel):
                bounds=self.patch_drag;self.patch_drag=None
                if self.doc.pointer_capture==self.relay.handle:self.doc.release_pointer_capture(self.relay.handle)
                if event.type==PointerEventType.Up:
                    self.app._safe(lambda:self.set_patch(bounds))
                self.app.composition.request_repaint()
            else:self.app.composition.request_repaint()
            return True
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
        self._cached_image_path=None;self._cached_image=None
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
        self.submit(lambda cancel:self.service.generate(self.root,settings,cancel,patch=self.patch),self.image_ready)

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
        self.preview_mask(show_mask=False)

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
        self.canvas.set_paint_callback(None)
        self.relay.set_pointer_handler(None)
        self.connections.clear()
