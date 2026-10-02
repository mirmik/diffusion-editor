"""Prepare accepted bald references as three masked head/short-neck crops.

Original image pixels are retained. Colour masking only separates the warm
skin from grey background/blue clothing; filled interior holes retain eyes.
These are nominal cameras, not a calibrated reconstruction dataset.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def main():
    root=Path(__file__).resolve().parents[2]
    references=root/'experiments/procedural-vaan/references/profile-hair-removal'
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
    frames=[];records=[]
    fov=math.radians(20);occupancy=.86
    distance=1/(2*math.tan(fov/2)*occupancy)
    for angle,name,bottom in [(0,'front.png',710),(90,'right-hairless.png',820),(270,'left-hairless.png',820)]:
        path=references/name
        im=Image.open(path).convert('RGB');rgb=np.asarray(im).astype(np.int16)
        skin=(rgb[:,:,0]-rgb[:,:,2]>25)&(rgb[:,:,0]-rgb[:,:,1]>7)
        skin[bottom:]=False
        mask=Image.fromarray(skin.astype(np.uint8)*255).copy()
        ImageDraw.floodfill(mask,(0,0),128)
        mask=Image.fromarray((np.asarray(mask)!=128).astype(np.uint8)*255)
        bbox=mask.getbbox();span=math.ceil((bbox[3]-bbox[1])/occupancy)
        x=round((bbox[0]+bbox[2]-span)/2);y=round((bbox[1]+bbox[3]-span)/2)
        crop=(x,y,x+span,y+span)
        mask=mask.filter(ImageFilter.GaussianBlur(.65))
        im=im.convert('RGBA');im.putalpha(mask)
        prepared=im.crop(crop).resize((1024,1024),Image.Resampling.LANCZOS)
        filename=f'view-{angle:03d}.png';prepared.save(out/filename)
        a=math.radians(angle);s,c=round(math.sin(a),10),round(math.cos(a),10)
        frames.append({'file_path':filename,'name':f'azim{angle:03d}',
            'transform_matrix':[[c,0,s,distance*s],[s,0,-c,-distance*c],[0,1,0,0],[0,0,0,1]]})
        records.append(dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bottom_cut_px=bottom,mask_bbox=bbox,square_crop=crop,output=filename,
            output_sha256=hashlib.sha256((out/filename).read_bytes()).hexdigest()))
    (out/'transforms.json').write_text(json.dumps(dict(camera_angle_x=fov,mesh_scale=1.0,frames=frames),indent=2)+'\n')
    (out/'preparation.json').write_text(json.dumps(dict(records=records,occupancy=occupancy,
        camera_assumption='Nominal cardinal views; independently height-normalized head with short neck. No calibrated poses.',
        limitations='Accepted generated profiles have uncertain scalp volume and altered brows; no back image.'),indent=2)+'\n')
    sys.path.insert(0,str(root/'experiments/procedural-vaan'))
    from preview import sheet
    previews=[]
    for f in frames:
        im=Image.open(out/f['file_path']).convert('RGBA');bg=Image.new('RGBA',im.size,'#737373');bg.alpha_composite(im)
        previews.append((f['name'],bg))
    sheet(previews,out/'inputs.jpg',3,'PIXAL3D / ACCEPTED BALD HEAD REFERENCES',380,380)
    request={'protocol':1,'backend':'pixal3d','operation':'shape','root':str(root/'.local/pixal3d-multiview/upstream'),
        'model_path':str(root/'.local/pixal3d-multiview/model'),'views_dir':'views',
        'settings':dict(seed=42,steps=12,resolution=1024,fov=20.0,normalize_views=True,decimation_target=100000,texture_size=1024)}
    (out.parent/'request.json').write_text(json.dumps(request,indent=2)+'\n')


if __name__=='__main__':main()
