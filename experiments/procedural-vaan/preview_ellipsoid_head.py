"""Show the ellipsoid experiment and its unchanged-camera baseline."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from preview import sheet


def main():
    out=Path(sys.argv[1])
    views=[('FRONT','face'),('PROFILE','face-side'),('THREE QUARTER','face-three-quarter')]
    sheet([(label,out/f'{view}.png') for label,view in views],out/'head-review.jpg',3,
          'VAAN / ELLIPSOID HEAD STUDY',450,450)
    sheet([(f'{prefix.upper()} / {label}',out/f'{file_prefix}{view}.png')
           for prefix,file_prefix in [('REVISION 3','baseline-'),('ELLIPSOIDS','')]
           for label,view in views],out/'before-after.jpg',3,
          'VAAN / SAME CAMERAS AND LIGHTING',450,450)
    sheet([('RAW POSITIVE ELLIPSOIDS',out/'construction.png'),
           ('SMOOTH UNION + SUBTRACTIONS',out/'face-three-quarter.png')],
          out/'construction-review.jpg',2,'VAAN / VOLUME CONSTRUCTION',550,550)


if __name__=='__main__':main()
