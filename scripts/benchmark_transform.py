"""Measure transform CPU preview work, excluding window rendering and upload."""

import argparse
import time

import numpy as np

from diffusion_editor.canvas.canvas_transform import CanvasTransformController
from diffusion_editor.canvas.editor_canvas_controller import EditorCanvasController
from diffusion_editor.document.document_service import DocumentService
from diffusion_editor.document.history import HistoryManager
from diffusion_editor.document.layer_stack import LayerStack


def benchmark(size, fragment, frames, multilayer=False, soft=False):
    stack = LayerStack()
    stack.init_from_image(np.full((size, size, 4), (150, 60, 30, 255), dtype=np.uint8))
    if multilayer:
        stack.add_layer("Subject", stack.active_layer.image.copy())
    stack.selection.data[100:100+fragment, 100:100+fragment] = .5 if soft else 1
    history = HistoryManager(stack.load_state)
    document = DocumentService(stack, history, stack.load_state)
    canvas = EditorCanvasController(stack, gpu_compositing=False,
        set_image=lambda _: None, set_overlay=lambda _: None,
        update_image_region=lambda *_: None)
    transform = CanvasTransformController(stack, document, canvas)
    try:
        transform.begin()
        start = 1200 if size == 2048 else 2500
        transform.set_rect((start, 100, start+fragment, 100+fragment))
        samples = []
        for i in range(1, frames+1):
            before = time.perf_counter()
            transform.set_rect((start+i, 100+i, start+i+fragment, 100+i+fragment))
            samples.append((time.perf_counter()-before)*1000)
        return float(np.median(samples))
    finally:
        transform.close()
        canvas.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--multilayer", action="store_true")
    parser.add_argument("--soft", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    for size, fragment in ((2048, 256), (4096, 512), (4096, 1024)):
        median = benchmark(size, fragment, args.frames, args.multilayer, args.soft)
        print(f"canvas={size}² fragment={fragment}² median={median:.2f} ms", flush=True)
