import threading
import time

import pytest

from diffusion_editor.generation.progress import GenerationProgress
from diffusion_editor.engines.threaded_lifecycle import EngineTaskQueue
from diffusion_editor.workers.ml_protocol import decode_response, encode_message, MlProtocolError
from diffusion_editor.workers.ml_backend import RealMlBackend
from diffusion_editor.workers.sensenova_image_edit import sampling_progress


@pytest.mark.parametrize("current,total", [(True, 8), (1, None), (None, 8), (-1, 8), (9, 8), (0, 0)])
def test_protocol_rejects_invalid_progress(current, total):
    with pytest.raises(MlProtocolError):
        decode_response(encode_message(dict(protocol=1, type="progress", request_id="job",
                                            message="Sampling", current=current, total=total)))


def test_protocol_supports_stage_messages_and_step_counts():
    for update in [GenerationProgress("Loading..."), GenerationProgress("Sampling", 3, 8)]:
        response = decode_response(encode_message(dict(protocol=1, type="progress", request_id="job",
                                                       **update.to_dict())))
        assert response.progress == update
    assert GenerationProgress("Sampling", 3, 8).fraction == .375


def test_progress_keeps_worker_busy_and_carries_the_terminal_job_identity():
    queue = EngineTaskQueue()
    released = threading.Event()
    emitted = threading.Event()
    def operation(cancel):
        queue.emit_progress(GenerationProgress("Sampling", 1, 2))
        emitted.set()
        released.wait(2)
        return "image"
    try:
        assert queue.submit("inference", operation, job_id="specific-job", name="progress-test")
        assert emitted.wait(1)
        update = queue.poll_event()
        assert update.job_id == "specific-job" and update.progress.current == 1
        assert queue.is_busy
        released.set()
        deadline = time.monotonic() + 2
        terminal = None
        while terminal is None and time.monotonic() < deadline:
            terminal = queue.poll_event()
            time.sleep(.001)
        assert terminal.result == "image" and terminal.progress is None
        assert terminal.job_id == update.job_id
        assert not queue.is_busy
    finally:
        released.set()
        queue.shutdown()


def test_diffusers_callback_uses_actual_shortened_schedule_and_preserves_tensors():
    updates = []
    class Pipe:
        num_timesteps = 3
        def __call__(self, num_inference_steps, callback_on_step_end=None):
            for i in range(self.num_timesteps):
                tensors = {"latents": object()}
                assert callback_on_step_end(self, i, 900-i, tensors) is tensors
            return "image"
    assert RealMlBackend._run_diffusers(Pipe(), {"num_inference_steps": 40}, updates.append) == "image"
    assert [(p.current, p.total) for p in updates if p.message == "Sampling"] == [(1,3), (2,3), (3,3)]
    assert updates[-1].message == "Decoding image..."


def test_sensenova_progress_is_step_based_and_restored_after_failure():
    class Model:
        def unpatchify(self, value):
            return value
    model = Model()
    updates = []
    with pytest.raises(RuntimeError):
        with sampling_progress(model, 2, updates.append):
            assert model.unpatchify("pixels") == "pixels"
            raise RuntimeError("cancelled")
    assert "unpatchify" not in vars(model)
    assert [(p.current, p.total) for p in updates] == [(0,2), (1,2)]
    assert model.unpatchify("unchanged") == "unchanged"
