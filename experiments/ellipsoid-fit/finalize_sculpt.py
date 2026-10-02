"""Persist packed references as real Blender datablocks, preserving build source."""
from pathlib import Path
import sys
import json
import hashlib
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from limits import require_budget
require_budget()
here=Path(__file__).resolve().parent
out=Path(bpy.data.filepath).parent
source=out/'source';source.mkdir(exist_ok=True)
(source/'sculpt.py').write_text(bpy.data.texts['sculpt.py'].as_string())
references={}
for filename in ('front.png','right-hairless.png','left-hairless.png'):
    path=here.parent/'procedural-vaan/references/profile-hair-removal'/filename
    img=bpy.data.images.load(str(path),check_existing=True)
    img.use_fake_user=True;img.pack()
    references[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
scene=bpy.context.scene;cam=scene.camera
cam.location=(1.7,-2.94,.07);cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler()
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_location=(0,0,0)
            area.spaces.active.region_3d.view_distance=1.8
            area.spaces.active.region_3d.view_rotation=cam.rotation_euler.to_quaternion()
            area.spaces.active.shading.type='MATERIAL'
(out/'reference-hashes.json').write_text(json.dumps(references,indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print('FINALIZED packed references',len(references),flush=True)
