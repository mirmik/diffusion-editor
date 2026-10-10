#!/usr/bin/env python3
"""Exercise ordered Text to Image references through the editor and real workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.tool import TextToImageTool, TextToImageReference
from diffusion_editor.engines.text_to_image_engine import TextToImageEngine
from diffusion_editor.generation.text_to_image_controller import TextToImageGenerationController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default='qwen-image-2.1-turbo')
    parser.add_argument('--output', type=Path, default=Path('/tmp/diffusion-editor-multiple-references'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    refs = []
    for index, (color, size) in enumerate([('red', (320, 192)), ('blue', (192, 320)), ('green', (256, 256))]):
        image = Image.new('RGBA', size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        w, h = size
        if index == 0:
            draw.polygon([(w//2, h//6), (w//6, h*5//6), (w*5//6, h*5//6)], fill=color)
        elif index == 1:
            draw.ellipse((w//6, h//4, w*5//6, h*3//4), fill=color)
        else:
            draw.rectangle((w//4, h//4, w*3//4, h*3//4), fill=color)
        image.save(args.output / f'input-{index+1}.png')
        refs.append(TextToImageReference(stable_id=f'ref-{index}', image=image))
    engine = TextToImageEngine()
    controller = TextToImageGenerationController(engine=engine)
    try:
        # Second run reorders, removes an input and changes dimensions without reload.
        for label, order, size in [('three', [0, 1, 2], (499, 384)), ('reordered-two', [2, 0], (384, 499))]:
            tool = TextToImageTool(args.profile, references=tuple(refs[i] for i in order))
            tool.set_parameter('seed', 123)
            tool.set_parameter('prompt',
                'Create a clean graphic poster on a white background. Arrange the shapes from '
                + ', '.join(f'<image{i+1}>' for i in range(len(order)))
                + ' side by side from left to right in that order, preserving their individual colors and shapes. No text.')
            layer = Layer('Generated', *size)
            layer.tool = tool
            start = controller.start(layer)
            if label == 'reordered-two':
                assert not start.model_loading, 'Reference changes should not reload weights'
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                event = controller.poll()
                if event:
                    assert not event.model_error and not event.inference_error, event
                    if event.inference_result:
                        context, image, seed = event.inference_result
                        assert image.size == size and image.mode == 'RGBA'
                        assert len(context.reference_images) == len(order)
                        provenance = context.result_provenance.to_dict()
                        recorded = provenance['request']['parameters']['references']
                        assert [r['stable_id'] for r in recorded] == [refs[i].stable_id for i in order]
                        assert [r['image_hash'] for r in recorded] == [r.content_hash for r in context.reference_images]
                        sampling = provenance['runtime']['provider_runtime']['sampling']
                        assert sampling['reference_count'] == len(order)
                        image.save(args.output / f'{label}.png')
                        (args.output / f'{label}.json').write_text(json.dumps(provenance, indent=2))
                        print(f'PASS {label}: {len(order)} references, {size}, seed={seed}', flush=True)
                        break
                time.sleep(.1)
            else:
                raise TimeoutError('Multiple reference smoke timed out')
    finally:
        engine.shutdown(5)


if __name__ == '__main__':
    main()
