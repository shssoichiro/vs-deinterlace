__all__ = ["QTGMCPresets", "QTGMCNoisePresets"]

from _collections_abc import dict_items, dict_values, dict_keys
from typing import TYPE_CHECKING, Any

from stgpytools import inject_self, KwargsT, KwargsNotNone, CustomEnum
from vsdenoise import SearchMode

from vsdeinterlace.qtgmc.enums import EdiMethod, DenoiseMethod, DeintMethod

if TYPE_CHECKING:

    class QTGMCPresetBase(dict[str, Any]): ...

else:
    QTGMCPresetBase = object


class QTGMCPreset(QTGMCPresetBase):
    """Base class for properties defined in a QTGMC preset"""

    if TYPE_CHECKING:

        def __call__(
            self,
            *,
            tr0: int,
            tr1: int,
            tr2_x: int,
            repair0: int,
            repair2: int,
            edi_mode: EdiMethod,
            nn_size: int,
            num_neurons: int,
            edi_max_dist: int,
            sharp_mode: int,
            sharp_limit_mode_x: int,
            sharp_limit_rad: int,
            sharp_back_blend: int,
            search_clip_pp: int,
            sub_pel: int,
            block_size: int,
            overlap: int,
            search: SearchMode,
            search_param: int,
            pel_search: int,
            chroma_motion: bool,
            precise: bool,
            prog_sad_mask: float
        ) -> QTGMCPreset: ...

    else:

        def __call__(self, **kwargs: Any) -> QTGMCPreset:
            return QTGMCPreset(**(dict(**self) | kwargs))

    def _get_dict(self) -> KwargsT:
        return KwargsNotNone(
            **{
                key: value.__get__(self) if isinstance(value, property) else value
                for key, value in (
                    self._value_ if isinstance(self, CustomEnum) else self
                ).__dict__.items()
            }
        )

    def __getitem__(self, key: str) -> Any:
        return self._get_dict()[key]

    def __class_getitem__(cls, key: str) -> Any:
        return cls()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._get_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._get_dict()

    def copy(self) -> QTGMCPreset:
        return QTGMCPreset(**self._get_dict())

    def keys(self) -> dict_keys[str, Any]:
        return self._get_dict().keys()

    def values(self) -> dict_values[str, Any]:
        return self._get_dict().values()

    def items(self) -> dict_items[str, Any]:
        return self._get_dict().items()

    def __eq__(self, other: Any) -> bool:
        return False

    @inject_self
    def as_dict(self) -> KwargsT:
        return KwargsT(**self._get_dict())


class QTGMCPresets:
    """Presets for QTGMC speed/quality tradeoff"""

    DRAFT = QTGMCPreset(
        tr0=0,
        tr1=1,
        tr2_x=0,
        repair0=0,
        repair2=0,
        edi_mode=EdiMethod.BOB,
        nn_size=4,
        num_neurons=0,
        edi_max_dist=4,
        sharp_mode=0,
        sharp_limit_mode_x=0,
        sharp_limit_rad=1,
        sharp_back_blend=0,
        search_clip_pp=0,
        sub_pel=1,
        block_size=32,
        overlap=32 // 4,
        search=SearchMode.ONETIME,
        search_param=1,
        pel_search=1,
        chroma_motion=False,
        precise=False,
        prog_sad_mask=0.0,
    )

    ULTRA_FAST = QTGMCPreset(
        tr0=1,
        tr1=1,
        tr2_x=0,
        repair0=0,
        repair2=3,
        edi_mode=EdiMethod.BWDIF,
        nn_size=4,
        num_neurons=0,
        edi_max_dist=4,
        sharp_mode=2,
        sharp_limit_mode_x=0,
        sharp_limit_rad=1,
        sharp_back_blend=0,
        search_clip_pp=1,
        sub_pel=1,
        block_size=32,
        overlap=32 // 4,
        search=SearchMode.ONETIME,
        search_param=1,
        pel_search=1,
        chroma_motion=False,
        precise=False,
        prog_sad_mask=0.0,
    )

    SUPER_FAST = QTGMCPreset(
        tr0=1,
        tr1=1,
        tr2_x=0,
        repair0=0,
        repair2=3,
        edi_mode=EdiMethod.NNEDI3,
        nn_size=4,
        num_neurons=0,
        edi_max_dist=4,
        sharp_mode=2,
        sharp_limit_mode_x=0,
        sharp_limit_rad=1,
        sharp_back_blend=0,
        search_clip_pp=1,
        sub_pel=1,
        block_size=32,
        overlap=32 // 4,
        search=SearchMode.ONETIME,
        search_param=1,
        pel_search=1,
        chroma_motion=False,
        precise=False,
        prog_sad_mask=0.0,
    )

    VERY_FAST = QTGMCPreset(
        tr0=1,
        tr1=1,
        tr2_x=0,
        repair0=0,
        repair2=4,
        edi_mode=EdiMethod.NNEDI3,
        nn_size=4,
        num_neurons=0,
        edi_max_dist=5,
        sharp_mode=2,
        sharp_limit_mode_x=2,
        sharp_limit_rad=1,
        sharp_back_blend=0,
        search_clip_pp=2,
        sub_pel=1,
        block_size=32,
        overlap=32 // 4,
        search=SearchMode.HEXAGON,
        search_param=1,
        pel_search=1,
        chroma_motion=False,
        precise=False,
        prog_sad_mask=0.0,
    )

    FASTER = QTGMCPreset(
        tr0=1,
        tr1=1,
        tr2_x=0,
        repair0=0,
        repair2=4,
        edi_mode=EdiMethod.NNEDI3,
        nn_size=4,
        num_neurons=0,
        edi_max_dist=6,
        sharp_mode=2,
        sharp_limit_mode_x=2,
        sharp_limit_rad=1,
        sharp_back_blend=0,
        search_clip_pp=2,
        sub_pel=1,
        block_size=32,
        overlap=32 // 2,
        search=SearchMode.HEXAGON,
        search_param=2,
        pel_search=1,
        chroma_motion=False,
        precise=False,
        prog_sad_mask=0.0,
    )

    FAST = QTGMCPreset(
        tr0=2,
        tr1=1,
        tr2_x=0,
        repair0=3,
        repair2=4,
        edi_mode=EdiMethod.NNEDI3,
        nn_size=5,
        num_neurons=0,
        edi_max_dist=6,
        sharp_mode=2,
        sharp_limit_mode_x=2,
        sharp_limit_rad=1,
        sharp_back_blend=0,
        search_clip_pp=2,
        sub_pel=1,
        block_size=16,
        overlap=32 // 2,
        search=SearchMode.HEXAGON,
        search_param=2,
        pel_search=1,
        chroma_motion=False,
        precise=False,
        prog_sad_mask=0.0,
    )

    MEDIUM = QTGMCPreset(
        tr0=2,
        tr1=1,
        tr2_x=1,
        repair0=3,
        repair2=4,
        edi_mode=EdiMethod.NNEDI3,
        nn_size=5,
        num_neurons=1,
        edi_max_dist=7,
        sharp_mode=2,
        sharp_limit_mode_x=2,
        sharp_limit_rad=1,
        sharp_back_blend=0,
        search_clip_pp=3,
        sub_pel=1,
        block_size=16,
        overlap=32 // 2,
        search=SearchMode.HEXAGON,
        search_param=2,
        pel_search=1,
        chroma_motion=False,
        precise=False,
        prog_sad_mask=10.0,
    )

    SLOW = QTGMCPreset(
        tr0=2,
        tr1=1,
        tr2_x=1,
        repair0=4,
        repair2=4,
        edi_mode=EdiMethod.NNEDI3,
        nn_size=6,
        num_neurons=1,
        edi_max_dist=7,
        sharp_mode=2,
        sharp_limit_mode_x=2,
        sharp_limit_rad=1,
        sharp_back_blend=0,
        search_clip_pp=3,
        sub_pel=2,
        block_size=16,
        overlap=32 // 2,
        search=SearchMode.HEXAGON,
        search_param=2,
        pel_search=2,
        chroma_motion=False,
        precise=False,
        prog_sad_mask=10.0,
    )

    SLOWER = QTGMCPreset(
        tr0=2,
        tr1=2,
        tr2_x=1,
        repair0=4,
        repair2=4,
        edi_mode=EdiMethod.NNEDI3,
        nn_size=6,
        num_neurons=1,
        edi_max_dist=8,
        sharp_mode=2,
        sharp_limit_mode_x=2,
        sharp_limit_rad=1,
        sharp_back_blend=1,
        search_clip_pp=3,
        sub_pel=2,
        block_size=16,
        overlap=32 // 2,
        search=SearchMode.HEXAGON,
        search_param=2,
        pel_search=2,
        chroma_motion=True,
        precise=False,
        prog_sad_mask=10.0,
    )

    VERY_SLOW = QTGMCPreset(
        tr0=2,
        tr1=2,
        tr2_x=2,
        repair0=4,
        repair2=4,
        edi_mode=EdiMethod.NNEDI3,
        nn_size=6,
        num_neurons=2,
        edi_max_dist=10,
        sharp_mode=2,
        sharp_limit_mode_x=2,
        sharp_limit_rad=1,
        sharp_back_blend=1,
        search_clip_pp=3,
        sub_pel=2,
        block_size=16,
        overlap=32 // 2,
        search=SearchMode.HEXAGON,
        search_param=2,
        pel_search=2,
        chroma_motion=True,
        precise=True,
        prog_sad_mask=10.0,
    )

    PLACEBO = QTGMCPreset(
        tr0=2,
        tr1=2,
        tr2_x=3,
        repair0=4,
        repair2=4,
        edi_mode=EdiMethod.NNEDI3,
        nn_size=6,
        num_neurons=4,
        edi_max_dist=12,
        sharp_mode=2,
        sharp_limit_mode_x=2,
        sharp_limit_rad=3,
        sharp_back_blend=3,
        search_clip_pp=3,
        sub_pel=2,
        block_size=16,
        overlap=32 // 2,
        search=SearchMode.UMH,
        search_param=2,
        pel_search=2,
        chroma_motion=True,
        precise=True,
        prog_sad_mask=10.0,
    )


class QTGMCNoisePreset(QTGMCPresetBase):
    """Base class for properties defined in a QTGMC noise preset"""

    if TYPE_CHECKING:

        def __call__(
            self,
            *,
            denoiser: DenoiseMethod,
            denoise_mc: bool,
            noise_tr: int,
            noise_deint: DeintMethod,
            stabilize_noise: bool
        ) -> QTGMCPreset: ...

    else:

        def __call__(self, **kwargs: Any) -> QTGMCPreset:
            return QTGMCPreset(**(dict(**self) | kwargs))

    def _get_dict(self) -> KwargsT:
        return KwargsNotNone(
            **{
                key: value.__get__(self) if isinstance(value, property) else value
                for key, value in (
                    self._value_ if isinstance(self, CustomEnum) else self
                ).__dict__.items()
            }
        )

    def __getitem__(self, key: str) -> Any:
        return self._get_dict()[key]

    def __class_getitem__(cls, key: str) -> Any:
        return cls()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._get_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._get_dict()

    def copy(self) -> QTGMCPreset:
        return QTGMCPreset(**self._get_dict())

    def keys(self) -> dict_keys[str, Any]:
        return self._get_dict().keys()

    def values(self) -> dict_values[str, Any]:
        return self._get_dict().values()

    def items(self) -> dict_items[str, Any]:
        return self._get_dict().items()

    def __eq__(self, other: Any) -> bool:
        return False

    @inject_self
    def as_dict(self) -> KwargsT:
        return KwargsT(**self._get_dict())


class QTGMCNoisePresets:
    """Presets for QTGMC denoising speed/quality tradeoff"""

    FASTER = QTGMCNoisePreset(
        denoiser=DenoiseMethod.FFT3DF,
        denoise_mc=False,
        noise_tr=0,
        noise_deint=DeintMethod.NONE,
        stabilize_noise=False,
    )

    FAST = QTGMCNoisePreset(
        denoiser=DenoiseMethod.FFT3DF,
        denoise_mc=False,
        noise_tr=1,
        noise_deint=DeintMethod.NONE,
        stabilize_noise=False,
    )

    MEDIUM = QTGMCNoisePreset(
        denoiser=DenoiseMethod.DFTTEST,
        denoise_mc=False,
        noise_tr=1,
        noise_deint=DeintMethod.NONE,
        stabilize_noise=True,
    )

    SLOW = QTGMCNoisePreset(
        denoiser=DenoiseMethod.DFTTEST,
        denoise_mc=True,
        noise_tr=1,
        noise_deint=DeintMethod.BOB,
        stabilize_noise=True,
    )

    SLOWER = QTGMCNoisePreset(
        denoiser=DenoiseMethod.DFTTEST,
        denoise_mc=True,
        noise_tr=2,
        noise_deint=DeintMethod.GENERATE,
        stabilize_noise=True,
    )
