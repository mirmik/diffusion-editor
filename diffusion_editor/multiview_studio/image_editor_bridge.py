"""Open the real editor for a captured camera image, with explicit return."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from .stablegen_service import fingerprint


def return_image(application, root):
    """Validate framing and save both editable document and flattened result."""
    pixels = application.layer_stack.composite()
    with Image.open(root/'editor-input.png') as source:
        expected = source.size
    if (pixels.shape[1], pixels.shape[0]) != expected:
        raise ValueError(f'Keep the captured canvas dimensions {expected} before returning to Studio')
    application.layer_stack.save_project(str(root/'image-edit.deproj'))
    Image.fromarray(pixels).convert('RGB').save(root/'editor-result.png')
    # Match the RGB serialization used when Studio imports the result.
    (root/'editor-return.json').write_text(json.dumps({
        'image_sha256': fingerprint(root/'editor-result.png'), 'status': 'returned'
    }))
    (root/'editor-document.json').write_text(json.dumps({
        'image_sha256': fingerprint(root/'editor-result.png')
    }))
    application.mark_document_saved(str(root/'image-edit.deproj'))
    application.request_stop()


def main(root):
    from diffusion_editor.app.application import EditorApplication
    from diffusion_editor.app.native_root import NativeEditorRoot
    from diffusion_editor.document.mask import Mask
    application = EditorApplication()
    with NativeEditorRoot.create_windowed(application, title='Edit camera image · return to Studio') as host:
        metadata=root/'editor-document.json'
        restore=(root/'image-edit.deproj').is_file() and metadata.is_file()
        if restore:
            restore=json.loads(metadata.read_text())['image_sha256']==fingerprint(root/'editor-input.png')
        if restore:
            host.dialog_coordinator.open_project_path(str(root/'image-edit.deproj'))
        else:
            host.dialog_coordinator.import_image_path(str(root/'editor-input.png'))
            layer=application.layer_stack.active_layer
            if layer is not None and (root/'mask.png').is_file():
                layer.mask=Mask.from_uint8(np.array(Image.open(root/'mask.png').convert('L')))
                layer.patch_rect=layer.mask.bbox()
        document=host.composition.document
        button=document.create_button('Return image to Studio (projection is a separate step)')
        button.widget.stable_id='studio.return-image'
        host.view.root.add_fixed_child(button.widget,32.)
        def send():
            try:
                if any(getattr(c,'pending_contexts',()) for c in application._generation_controllers()):
                    application.set_status('Wait for image generation to finish before returning')
                    return
                return_image(application,root)
            except Exception as error:
                application.set_status(str(error))
        connection=button.connect_clicked(send)
        application.set_status('Use Patch and any available model. Return image when ready; keep canvas size and framing.')
        while application.running:
            if (root/'editor-cancel').exists():
                application.request_stop()
                break
            result=host.tick()
            if not result.rendered and not result.events and not result.dispatched:
                time.sleep(.01)
        # NativeEditorRoot closes the application and its workers before exit.
        del connection


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
