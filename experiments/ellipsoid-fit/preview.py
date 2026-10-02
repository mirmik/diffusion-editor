"""Contact sheets of identical-camera target, initial and fitted head renders."""
from pathlib import Path
import sys
import json
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'procedural-vaan'))
from preview import sheet

out=Path(sys.argv[1]);views=[('FRONT','front'),('PROFILE','profile'),('THREE QUARTER','three-quarter')]
params=json.loads((out/'fitted.json').read_text())
stages=params.get('stages',[dict(params,operation='union')])
positive=sum(len(s['parts']) for s in stages if s['operation']=='union')
negative=sum(len(s['parts']) for s in stages if s['operation']=='subtract')
sheet([(f'{stage.upper()} / {label}',out/f'{stage}-{view}.png')
       for stage in ['target','initial','fitted'] for label,view in views],
      out/'comparison.jpg',3,'PIXAL3D VOLUME / INITIAL / FITTED ELLIPSOIDS',400,400)
sheet([('PIXAL3D VOLUME',out/'target-three-quarter.png'),
       (f'{positive} ADD / {negative} CUT',out/'ellipsoids.png'),
       ('FITTED VOLUME',out/'fitted-three-quarter.png')],
      out/'result.jpg',3,'ELLIPSOID FIT / GEOMETRY ONLY',450,450)
sheet([(name.upper(),Image.open(out/f'{name}-front.png').crop((150,300,650,730)))
       for name in ['target','initial','fitted']],out/'face-detail.jpg',3,
      'FACE DETAIL / TARGET - BEFORE - AFTER',480,430)
