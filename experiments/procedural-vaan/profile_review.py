"""Compare actual saved head meshes and unchanged reference/render pixels."""
import argparse
import json
from pathlib import Path
import sys
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from preview import sheet


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--bare-baseline',type=Path,required=True)
    parser.add_argument('--anatomy-references',type=Path)
    args=parser.parse_args()
    p,old=args.run,args.baseline
    refs=json.loads((p/'report.json').read_text())['references']
    items=[]
    for key,view,box in [('right','face-side',(478,43,725,290)),
                         ('left','face-left',(452,43,699,290))]:
        crop=Image.open(refs[key]['path']).crop(box)
        items.extend([(f'{key.upper()} / REFERENCE',crop),
                      ('REVISION 2',old/f'{view}.png'),
                      ('REVISION 3',p/f'{view}.png')])
    sheet(items,p/'profile-reference-review.jpg',3,
          'VAAN / BOTH PROFILES / REFERENCE - BEFORE - AFTER',480,480)
    sheet([('REVISION 2',old/'face-side.png'),('REVISION 3',p/'face-side.png')],
          p/'before-after-profile.jpg',2,'VAAN / PROFILE / IDENTICAL CAMERAS',600,600)
    diagnostic=p/'diagnostic'
    before=json.loads((args.bare_baseline/'head-measurements.json').read_text())
    after=json.loads((diagnostic/'head-measurements.json').read_text())
    keys=['maximum_depth_mm','height_mm','maximum_width_including_ears_mm']
    measurements={'baseline':str(old),'revised':str(p),
                  'measurement':'World-space bounds of actual saved skin head mesh, excluding hair',
                  'before':{k:before[k] for k in keys},'after':{k:after[k] for k in keys},
                  'difference_mm':{k:after[k]-before[k] for k in keys}}
    (p/'head-depth-comparison.json').write_text(json.dumps(measurements,indent=2)+'\n')
    sheet([(f'BEFORE / DEPTH {before[keys[0]]:.1f} MM',args.bare_baseline/'bare-face-side.png'),
           (f'AFTER / DEPTH {after[keys[0]]:.1f} MM',diagnostic/'bare-face-side.png'),
           ('BEFORE / FRONT',args.bare_baseline/'bare-face.png'),
           ('AFTER / FRONT',diagnostic/'bare-face.png')],p/'bare-head-progress.jpg',2,
          'VAAN / SKULL GEOMETRY / HAIR HIDDEN',520,520)
    if args.anatomy_references:
        a=args.anatomy_references
        # Only uniform framing changes for references; no image retouching.
        refs=[('FRONT',a/'front.png','face'),
              ('RIGHT',a/'right-hairless.png','face-side'),
              ('LEFT',a/'left-hairless.png','face-left')]
        bare_items=[(f'{label} / '+('ORIGINAL CROP' if label=='FRONT' else 'HAIR REMOVAL'),ref)
                    for label,ref,view in refs]
        bare_items.extend((f'{label} / PROCEDURAL',diagnostic/f'bare-{view}.png') for label,ref,view in refs)
        sheet(bare_items,p/'anatomy-reference-review.jpg',3,
              'VAAN / HAIR-FREE REFERENCES & GEOMETRY / APPROXIMATE FRAMING',430,430)
    print(json.dumps(measurements,indent=2))


if __name__=='__main__':
    main()
