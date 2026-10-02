"""Use the installed Qwen editor to remove hair from existing profile views.

Run with project-venv Python and PYTHONPATH=.; the isolated ML worker uses its
own existing environment. Only the default Lightning LoRA is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
from PIL import Image
from diffusion_editor.generation.image_edit_profiles import QWEN_IMAGE_EDIT_PROFILE_ID,image_edit_profile
from diffusion_editor.workers.ml_process import MlProcessClient

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from preview import sheet

PROMPT=('Remove the white scalp hair completely, making this same character bald. '
        'Reveal a natural smooth scalp and the ear where the hair was. '
        'Preserve the exact existing side-view camera angle and head pose. '
        'Preserve this exact character identity, face silhouette, nose, lips, '
        'chin, eyes, eyebrows, ear position, neck and expression. '
        'Keep the eyebrows. Preserve the blue clothing, background, framing, '
        'lighting, skin colour and illustration style. Do not rotate the head. '
        'Do not redesign or beautify the face. Change only the scalp hair.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261003)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for name,path,box in [('right','/home/mirmik/Vaan/views/mv-eye-090.png',(478,43,725,290)),
                          ('left','/home/mirmik/Vaan/views/mv-eye-270.png',(452,43,699,290))]:
        source=Path(path)
        crop=Image.open(source).convert('RGB').crop(box)
        crop.save(out/f'{name}-original-crop.png')
        image=crop.resize((1024,1024),Image.Resampling.LANCZOS)
        image.save(out/f'{name}-input.png')
        jobs.append({'name':name,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'crop_xyxy':box,'input':f'{name}-input.png','output':f'{name}-hairless.png'})
    # The user-provided front is retained; no generated rotation is reused.
    front=Path('/home/mirmik/Vaan/Body.png')
    shutil.copy2(front,out/'Body.png')
    Image.open(front).crop((440,46,724,330)).resize((1024,1024),Image.Resampling.LANCZOS).save(out/'front.png')
    profile=image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    parameters=profile.defaults()
    parameters.update(seed=args.seed,steps=4,prompt=PROMPT)
    adapters=[a.to_dict() for a in profile.default_lora_adapters]
    manifest={'operation':'remove hair from pre-existing profile references; preserve pose and identity',
              'profile':profile.stable_id,'parameters':parameters,'lora_adapters':adapters,
              'multiangle_lora':False,'jobs':jobs,'status':'pending'}
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    client=MlProcessClient()
    cancel=threading.Event()
    started=time.monotonic()
    try:
        loaded=client.request('load_image_edit',{'profile_id':profile.stable_id,'parameters':parameters,'lora_adapters':adapters},
                              cancel,on_progress=lambda x:print(x,flush=True))
        manifest['loaded']=loaded
        save()
        for job in jobs:
            print('Editing',job['name'],flush=True)
            result=client.request('image_edit',{'profile_id':profile.stable_id,'parameters':parameters,'lora_adapters':adapters},
                                  cancel,images={'image':Image.open(out/job['input']).convert('RGB')},
                                  on_progress=lambda x:print(x,flush=True))
            image=result['image']
            image.save(out/job['output'])
            job['output_size']=list(image.size)
            job['output_sha256']=hashlib.sha256((out/job['output']).read_bytes()).hexdigest()
            save()
    finally:
        client.shutdown()
    manifest.update(status='generated_pending_visual_review',elapsed_seconds=time.monotonic()-started)
    save()
    sheet([(f'{j["name"].upper()} / ORIGINAL PROFILE',out/j['input']) if i==0 else
           (f'{j["name"].upper()} / HAIR REMOVAL',out/j['output'])
           for j in jobs for i in (0,1)],out/'hair-removal-review.jpg',2,
          'VAAN / EXISTING PROFILES / QWEN HAIR REMOVAL',520,520)
    print('DONE',out,manifest['elapsed_seconds'],flush=True)


if __name__=='__main__':
    main()
