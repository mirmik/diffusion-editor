"""Settings for diffusion-editor (backed by termin.base.Settings)."""

from termin.base.settings import Settings as _TcSettings


class Settings(_TcSettings):
    """App settings with auto-save on set()."""

    def __init__(self):
        super().__init__("diffusion-editor")

    def set(self, key: str, value):
        super().set(key, value)
        self.save()
