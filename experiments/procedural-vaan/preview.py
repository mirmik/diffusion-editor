"""Build compact reference comparisons with project-venv Pillow."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps


def sheet(items, path, columns, title, width=400, height=520):
    rows=(len(items)+columns-1)//columns
    out=Image.new('RGB',(columns*width,80+rows*(height+40)),'#171c23')
    draw=ImageDraw.Draw(out)
    font=lambda s:ImageFont.truetype('DejaVuSans.ttf',s)
    draw.text((20,18),title,font=font(23),fill='#f0e9dc')
    draw.text((20,51),'Vaan / procedural geometry / Blender study',font=font(14),fill='#96a4b2')
    for i,(label,file) in enumerate(items):
        x,y=(i%columns)*width,80+(i//columns)*(height+40)
        image=(file if isinstance(file,Image.Image) else Image.open(file)).convert('RGB')
        image=ImageOps.contain(image,(width,height),Image.Resampling.LANCZOS)
        out.paste(image,(x+(width-image.width)//2,y+40+(height-image.height)//2))
        draw.text((x+16,y+10),label,font=font(16),fill='#e2e5e9')
    out.save(path,quality=93)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--baseline',type=Path,help='Earlier run with the same orthographic cameras')
    args=parser.parse_args()
    p=args.run
    report=json.loads((p/'report.json').read_text())
    refs={k:Path(v['path']) for k,v in report['references'].items()}
    sheet([('REFERENCE / FRONT',refs['front']),('PROCEDURAL / FRONT',p/'front.png'),('PROCEDURAL / 3-QUARTER',p/'three-quarter.png')],p/'comparison.jpg',3,'VAAN / REFERENCE & PROCEDURAL STUDY')
    sheet([(k.upper(),p/f'{k}.png') for k in ['front','right','back','left']],p/'turnaround.jpg',4,'VAAN / FOUR VIEWS',350,500)
    sheet([('REFERENCE / RIGHT',refs['right']),('PROCEDURAL / RIGHT',p/'right.png'),('REFERENCE / BACK',refs['back']),('PROCEDURAL / BACK',p/'back.png')],p/'side-comparison.jpg',4,'VAAN / PROFILE & BACK REFERENCES',350,500)
    sheet([('FRONT DETAIL',p/'face.png'),('PROFILE DETAIL',p/'face-side.png')],p/'face-review.jpg',2,'VAAN / FACE GEOMETRY',500,500)
    if (p/'face-left.png').exists():
        sheet([(label,p/f'{name}.png') for label,name in [('FRONT','face'),('THREE QUARTER','face-three-quarter'),('RIGHT PROFILE','face-side'),('LEFT PROFILE','face-left'),('BACK HAIR','head-back')]],p/'portrait-turnaround.jpg',5,'VAAN / PORTRAIT VIEWS',400,400)
    if args.baseline:
        old=args.baseline
        sheet([('PREVIOUS PASS',old/'face.png'),('REVISED FACE & HAIR',p/'face.png')],p/'before-after-face.jpg',2,'VAAN / SAME CAMERA COMPARISON',600,600)
        sheet([('PREVIOUS PASS',old/'three-quarter.png'),('REVISED MODEL',p/'three-quarter.png')],p/'before-after-body.jpg',2,'VAAN / SAME CAMERA COMPARISON',500,670)
        # Manual framing only: unchanged reference pixels, no image synthesis.
        front=Image.open(refs['front']).crop((459,41,706,288))
        profile=Image.open(refs['right']).crop((478,43,725,290))
        sheet([('FRONT REFERENCE',front),('PREVIOUS FRONT',old/'face.png'),('REVISED FRONT',p/'face.png'),('PROFILE REFERENCE',profile),('PREVIOUS PROFILE',old/'face-side.png'),('REVISED PROFILE',p/'face-side.png')],p/'likeness-review.jpg',3,'VAAN / REFERENCE - BEFORE - AFTER',480,480)


if __name__=='__main__':
    main()
