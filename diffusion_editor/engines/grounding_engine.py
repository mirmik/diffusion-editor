"""Grounding DINO + SAM 2.1 backend adapter."""

from __future__ import annotations

from queue import Empty, Queue
from threading import Thread
import time

import numpy as np
from tcbase import log

from ..grounding.types import (
    GroundingDetection,
    GroundingEngineEvent,
    GroundingRequest,
    GroundingResult,
)


def _import_torch():
    import torch
    return torch


class GroundingEngine:
    def __init__(self):
        self._busy = False
        self._thread: Thread | None = None
        self._events: Queue[GroundingEngineEvent] = Queue()
        self._dino_model = None
        self._dino_proc = None
        self._dino_model_id: str | None = None
        self._dino_device: str | None = None
        self._sam2_model = None
        self._sam2_proc = None
        self._sam2_model_id: str | None = None
        self._sam2_device: str | None = None

    @property
    def is_busy(self) -> bool:
        return self._busy

    def gpu_available(self) -> bool:
        try:
            torch = _import_torch()
            return bool(torch.cuda.is_available())
        except Exception as exc:
            log.error(f"Grounding: failed to query GPU availability: {exc}")
            return False

    def submit_request(self, request: GroundingRequest) -> bool:
        if self._busy:
            return False
        self._busy = True
        self._drain_events()
        self._thread = Thread(target=self._run, args=(request,), daemon=True)
        self._thread.start()
        return True

    def poll_event(self) -> GroundingEngineEvent | None:
        try:
            return self._events.get_nowait()
        except Empty:
            return None

    def _drain_events(self) -> None:
        while True:
            try:
                self._events.get_nowait()
            except Empty:
                return

    def _status(self, message: str) -> None:
        self._events.put(GroundingEngineEvent(status=message))

    def _run(self, request: GroundingRequest) -> None:
        params = request.params
        arr = request.image
        t0 = time.time()

        try:
            log.info(
                f"Grounding start: model={params.model_id}, "
                f"prompt='{params.prompt}', "
                f"box_threshold={params.box_threshold:.2f}, "
                f"text_threshold={params.text_threshold:.2f}, "
                f"sam2={params.sam2_model_id}, "
                f"mask_channel={params.sam2_mask_channel}, gpu={params.use_gpu}, "
                f"mask_threshold={params.mask_threshold:.2f}, "
                f"max_hole={params.max_hole_area}, "
                f"max_sprinkle={params.max_sprinkle_area}, "
                f"multimask={params.multimask}, "
                f"non_overlap={params.non_overlap}, "
                f"image={arr.shape[1]}x{arr.shape[0]}",
            )

            model, proc = self._get_dino_model(
                params.model_id,
                params.use_gpu,
                status_fn=self._status,
            )
            t1 = time.time()
            log.info(f"Grounding: model ready in {t1 - t0:.1f}s")

            self._status("Grounding: detecting...")
            boxes = self._run_grounding(
                proc,
                model,
                arr,
                params.prompt,
                params.box_threshold,
                params.text_threshold,
                params.use_gpu,
            )
            t2 = time.time()
            log.info(f"Grounding: inference done in {t2 - t1:.1f}s")

            if not boxes:
                log.info(f"Grounding: nothing found for prompt='{params.prompt}'")
                self._events.put(
                    GroundingEngineEvent(status="Grounding: nothing found")
                )
                return

            masks = [None] * len(boxes)
            if params.sam2_model_id:
                try:
                    self._status("SAM 2.1: segmenting...")
                    sam_model, sam_proc = self._get_sam2_model(
                        params.sam2_model_id,
                        params.use_gpu,
                        status_fn=self._status,
                    )
                    t_sam0 = time.time()
                    masks = self._run_sam2(
                        sam_proc,
                        sam_model,
                        arr,
                        boxes,
                        params.sam2_mask_channel,
                        params.mask_threshold,
                        params.max_hole_area,
                        params.max_sprinkle_area,
                        params.multimask,
                        params.non_overlap,
                        params.use_gpu,
                    )
                    t_sam1 = time.time()
                    log.info(
                        f"SAM 2.1: {sum(1 for m in masks if m is not None)} "
                        f"masks in {t_sam1 - t_sam0:.1f}s"
                    )
                except Exception as exc:
                    log.error(f"SAM 2.1: {exc}")
                    self._events.put(
                        GroundingEngineEvent(status=f"SAM 2.1 error: {exc}")
                    )

            detections = tuple(
                GroundingDetection(
                    label=label,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    score=score,
                    mask=mask,
                )
                for (label, x0, y0, x1, y1, score), mask in zip(boxes, masks)
            )
            found = ", ".join(
                f"{item.label} ({item.score:.0%})" for item in detections
            )
            log.info(
                f"Grounding: {len(detections)} hit(s) - {found} "
                f"(total {time.time() - t0:.1f}s)"
            )
            self._events.put(
                GroundingEngineEvent(
                    status=f"Grounding: {len(detections)} hit(s): {found}",
                    result=GroundingResult(detections=detections),
                )
            )
        except Exception as exc:
            log.error(f"Grounding: {exc}")
            self._events.put(GroundingEngineEvent(error=str(exc)))
        finally:
            self._busy = False

    def _get_dino_model(self, model_id: str, use_gpu: bool, status_fn=None):
        device = "cuda" if use_gpu else "cpu"
        log.info(f"Grounding: loading DINO model={model_id}, device={device}")

        if self._dino_model is not None and model_id != self._dino_model_id:
            log.info(
                "Grounding: DINO model changed "
                f"{self._dino_model_id} -> {model_id}, resetting cache"
            )
            self._dino_model = None
            self._dino_proc = None
            self._dino_device = None

        if self._dino_model is None:
            from transformers import (
                GroundingDinoForObjectDetection,
                GroundingDinoProcessor,
            )
            from huggingface_hub import try_to_load_from_cache

            t_load = time.time()
            cached = try_to_load_from_cache(model_id, "model.safetensors") is not None
            log.info(
                f"Grounding: DINO cache check for {model_id} -> cached={cached} "
                f"({time.time() - t_load:.1f}s)"
            )

            if status_fn:
                short_name = model_id.split("/")[-1]
                if cached:
                    status_fn(f"Loading {short_name} from cache...")
                else:
                    status_fn(
                        f"Downloading {short_name} (this may take a few minutes)..."
                    )

            self._dino_proc = GroundingDinoProcessor.from_pretrained(
                model_id, local_files_only=cached)
            self._dino_model = GroundingDinoForObjectDetection.from_pretrained(
                model_id, local_files_only=cached)
            self._dino_model_id = model_id
            self._dino_device = None
            log.info(f"Grounding: DINO model loaded in {time.time() - t_load:.1f}s")

        if self._dino_device != device:
            t_move = time.time()
            self._dino_model.to(device)
            self._dino_model.eval()
            log.info(
                f"Grounding: moved DINO model to {device} in "
                f"{time.time() - t_move:.1f}s"
            )
            self._dino_device = device

        return self._dino_model, self._dino_proc

    def _run_grounding(
            self,
            processor,
            model,
            image_array,
            prompt: str,
            box_threshold: float,
            text_threshold: float,
            use_gpu: bool):
        from PIL import Image

        device = "cuda" if use_gpu else "cpu"

        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array, "RGBA").convert("RGB")
        else:
            image = image_array

        log.info(
            f"Grounding: running inference, image={image.width}x{image.height}, "
            f"prompt='{prompt}', box_threshold={box_threshold:.2f}, "
            f"text_threshold={text_threshold:.2f}, device={device}"
        )

        t_prep = time.time()
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        log.info(f"Grounding: inputs prepared in {time.time() - t_prep:.1f}s")

        t_infer = time.time()
        torch = _import_torch()
        with torch.no_grad():
            outputs = model(**inputs)
        log.info(f"Grounding: model forward pass in {time.time() - t_infer:.1f}s")

        h, w = image.height, image.width
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )

        boxes = []
        for result in results:
            for score, box, label in zip(
                result["scores"].tolist(),
                result["boxes"].tolist(),
                result["labels"],
            ):
                x0, y0, x1, y1 = [int(value) for value in box]
                boxes.append((label, x0, y0, x1, y1, score))
                log.info(
                    f"Grounding: detected '{label}' box=({x0},{y0},{x1},{y1}) "
                    f"score={score:.3f}"
                )

        boxes.sort(key=lambda box: box[5], reverse=True)
        log.info(f"Grounding: post-processing done, {len(boxes)} boxes found")
        return boxes

    def _get_sam2_model(self, model_id: str, use_gpu: bool, status_fn=None):
        device = "cuda" if use_gpu else "cpu"
        log.info(f"SAM 2.1: loading model={model_id}, device={device}")

        if self._sam2_model is not None and model_id != self._sam2_model_id:
            log.info(
                f"SAM 2.1: model changed {self._sam2_model_id} -> {model_id}, "
                "resetting cache"
            )
            self._sam2_model = None
            self._sam2_proc = None
            self._sam2_device = None

        if self._sam2_model is None:
            from transformers import Sam2Processor, Sam2Model
            from huggingface_hub import try_to_load_from_cache

            t_load = time.time()
            cached = try_to_load_from_cache(model_id, "model.safetensors") is not None
            log.info(
                f"SAM 2.1: cache check for {model_id} -> cached={cached} "
                f"({time.time() - t_load:.1f}s)"
            )

            if status_fn:
                short_name = model_id.split("/")[-1]
                if cached:
                    status_fn(f"Loading {short_name} from cache...")
                else:
                    status_fn(
                        f"Downloading {short_name} (this may take a few minutes)..."
                    )

            self._sam2_proc = Sam2Processor.from_pretrained(
                model_id, local_files_only=cached)
            self._sam2_model = Sam2Model.from_pretrained(
                model_id, local_files_only=cached)
            self._sam2_model_id = model_id
            self._sam2_device = None
            log.info(f"SAM 2.1: model loaded in {time.time() - t_load:.1f}s")

        if self._sam2_device != device:
            t_move = time.time()
            self._sam2_model.to(device)
            self._sam2_model.eval()
            log.info(
                f"SAM 2.1: moved model to {device} in "
                f"{time.time() - t_move:.1f}s"
            )
            self._sam2_device = device

        return self._sam2_model, self._sam2_proc

    def _run_sam2(
            self,
            processor,
            model,
            image_array,
            boxes,
            mask_channel,
            mask_threshold,
            max_hole_area,
            max_sprinkle_area,
            multimask,
            non_overlap,
            use_gpu):
        from PIL import Image

        device = "cuda" if use_gpu else "cpu"

        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array, "RGBA").convert("RGB")
        else:
            image = image_array

        h, w = image.height, image.width
        masks = []
        input_boxes = [[[x0, y0, x1, y1] for _, x0, y0, x1, y1, _ in boxes]]

        t_prep = time.time()
        inputs = processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        log.info(
            f"SAM 2.1: inputs prepared in {time.time() - t_prep:.1f}s "
            f"({len(boxes)} boxes)"
        )

        t_infer = time.time()
        torch = _import_torch()
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=multimask)
        log.info(f"SAM 2.1: forward pass in {time.time() - t_infer:.1f}s")

        masks_tensor = processor.post_process_masks(
            outputs.pred_masks,
            original_sizes=[(h, w)],
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
            apply_non_overlapping_constraints=non_overlap,
            binarize=True,
        )
        masks_tensor = masks_tensor[0]

        for index in range(len(boxes)):
            mask_tensor = masks_tensor[index]
            if multimask and mask_tensor.dim() > 2:
                mask_tensor = mask_tensor[mask_channel]
            mask = mask_tensor.cpu().numpy().astype(bool)
            masks.append(mask)
            log.info(
                f"SAM 2.1: mask {index} area={mask.sum()}px "
                f"({mask.sum() / (h * w) * 100:.1f}% of image)"
            )

        return masks

    def unload(self) -> None:
        self._dino_model = None
        self._dino_proc = None
        self._dino_model_id = None
        self._dino_device = None
        self._sam2_model = None
        self._sam2_proc = None
        self._sam2_model_id = None
        self._sam2_device = None

    def shutdown(self, timeout: float = 1.0) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self.unload()
