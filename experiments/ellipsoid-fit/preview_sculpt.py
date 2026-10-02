"""Comparison sheets from Blender renders and unchanged reference crops."""
from pathlib import Path
import argparse
import sys
from PIL import Image
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'procedural-vaan'))
from preview import sheet

p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();out=args.run
views=['front','profile','three-quarter']
sheet([(f'{mode.upper()} / {v.upper()}',out/f'{mode}-{v}.png') for mode in ['base','clay'] for v in views],
      out/'before-after.jpg',3,'VAAN / ELLIPSOID BASE -> SURFACE SCULPT',450,450)
sheet([(v.upper(),out/f'skin-{v}.png') for v in views],out/'result.jpg',3,'VAAN / SURFACE SCULPT',450,450)
refs=HERE.parent/'procedural-vaan/references/profile-hair-removal'
# Framing crops retain original reference pixels; these are not image edits.
front=Image.open(refs/'front.png').crop((255,110,765,695))
profile=Image.open(refs/'right-hairless.png').crop((175,80,825,810))
front_model=Image.open(out/'skin-front.png');profile_model=Image.open(out/'skin-profile.png')
def crop_model(image):
    w,h=image.size;return image.crop((int(.12*w),int(.05*h),int(.89*w),int(.93*h)))
sheet([('FRONT REFERENCE',front),('SCULPT / FRONT',crop_model(front_model)),
       ('PROFILE REFERENCE',profile),('SCULPT / PROFILE',crop_model(profile_model))],
      out/'reference-review.jpg',4,'VAAN / REFERENCE & SCULPT',360,460)
for path in out.glob('*.png'):
    with Image.open(path) as image:image.verify()
for path in out.glob('*.jpg'):
    with Image.open(path) as image:image.verify()
print('Comparison sheets saved and images decoded',out)
