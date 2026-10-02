"""Read a saved scene, measure its head and render hair-free diagnostic views.

This never saves the modified Blender scene. Use --output outside the source
artifact directory when inspecting an earlier version.
"""
import argparse
import json
from pathlib import Path
import sys
import bpy
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from build import view


HAIR_NAMES=('Continuous silver','Swept silver','Temple and nape','Fine tapered nape',
            'Long back hair','Back crown','Asymmetric fringe','Fringe split',
            'Fine crown cowlick','Swept crown wisp','Directional silver')


def measure_head():
    obj=bpy.data.objects['Skin | head ears nose']
    p=np.empty(len(obj.data.vertices)*3,dtype=np.float64)
    obj.data.vertices.foreach_get('co',p)
    p=p.reshape(-1,3)
    transform=np.array(obj.matrix_world)
    p=p@transform[:3,:3].T+transform[:3,3]
    lower,upper=p.min(0),p.max(0)
    sections=[]
    for z in np.arange(1.57,1.781,.001):
        band=p[np.abs(p[:,2]-z)<.00065]
        if len(band):
            sections.append({'z':float(z),'front_y':float(band[:,1].min()),
                             'back_y':float(band[:,1].max()),
                             'half_width':float(np.abs(band[:,0]).max())})
    return {'source_blend':bpy.data.filepath,'bounds_metres':[lower.tolist(),upper.tolist()],
            'height_mm':float((upper[2]-lower[2])*1000),
            'maximum_depth_mm':float((upper[1]-lower[1])*1000),
            'maximum_width_including_ears_mm':float((upper[0]-lower[0])*1000),
            'horizontal_sections':sections}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--height',type=int,default=1000)
    parser.add_argument('--samples',type=int,default=40)
    parser.add_argument('--views',nargs='+',default=['face-side','face','face-three-quarter'])
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True,exist_ok=True)
    result=measure_head()
    (args.output/'head-measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    for obj in bpy.context.scene.objects:
        if obj.get('character_region')=='hair' or obj.name.startswith(HAIR_NAMES):
            obj.hide_render=True
    scene=bpy.context.scene
    scene.cycles.samples=args.samples
    for name in args.views:
        view(scene.camera,name,args.height)
        scene.render.filepath=str((args.output/f'bare-{name}.png').resolve())
        bpy.ops.render.render(write_still=True)
    print('HEAD MEASUREMENTS',json.dumps({k:v for k,v in result.items() if k!='horizontal_sections'}),flush=True)


if __name__=='__main__':
    main()
