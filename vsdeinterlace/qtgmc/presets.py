__all__ = ["QTGMCPreset", "QTGMCPresets", "QTGMCNoisePreset", "QTGMCNoisePresets"]

from typing import Any

from vsdenoise import SearchMode
from vsdeinterlace.qtgmc.enums import EdiMethod, DenoiseMethod, DeintMethod


class QTGMCPreset(dict[str, Any]):
    """Base class for properties defined in a QTGMC preset"""

    preset: int
    tr0: int
    tr1: int
    tr2: int
    repair0: int
    repair2: int
    edi_mode: EdiMethod
    nn_size: int
    num_neurons: int
    edi_max_dist: int
    sharp_mode: int
    sharp_limit_mode: int
    sharp_limit_rad: int
    sharp_back_blend: int
    search_clip_pp: int
    sub_pel: int
    block_size: int
    overlap: int
    search: SearchMode
    search_param: int
    pel_search: int
    chroma_motion: bool
    precise: bool
    prog_sad_mask: float


class QTGMCPresets:
    """Presets for QTGMC speed/quality tradeoff"""

    DRAFT = QTGMCPreset(
        preset=10,
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
        preset=9,
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
        preset=8,
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
        preset=7,
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
        preset=6,
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
        preset=5,
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
        preset=4,
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
        preset=3,
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
        preset=2,
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
        preset=1,
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
        preset=0,
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


class QTGMCNoisePreset(dict[str, Any]):
    """Base class for properties defined in a QTGMC noise preset"""

    preset: int
    denoiser: DenoiseMethod
    denoise_mc: bool
    noise_tr: int
    noise_deint: DeintMethod
    stabilize_noise: bool


class QTGMCNoisePresets:
    """Presets for QTGMC denoising speed/quality tradeoff"""

    FASTER = QTGMCNoisePreset(
        preset=4,
        denoiser=DenoiseMethod.FFT3DF,
        denoise_mc=False,
        noise_tr=0,
        noise_deint=DeintMethod.NONE,
        stabilize_noise=False,
    )

    FAST = QTGMCNoisePreset(
        preset=3,
        denoiser=DenoiseMethod.FFT3DF,
        denoise_mc=False,
        noise_tr=1,
        noise_deint=DeintMethod.NONE,
        stabilize_noise=False,
    )

    MEDIUM = QTGMCNoisePreset(
        preset=2,
        denoiser=DenoiseMethod.DFTTEST,
        denoise_mc=False,
        noise_tr=1,
        noise_deint=DeintMethod.NONE,
        stabilize_noise=True,
    )

    SLOW = QTGMCNoisePreset(
        preset=1,
        denoiser=DenoiseMethod.DFTTEST,
        denoise_mc=True,
        noise_tr=1,
        noise_deint=DeintMethod.BOB,
        stabilize_noise=True,
    )

    SLOWER = QTGMCNoisePreset(
        preset=0,
        denoiser=DenoiseMethod.DFTTEST,
        denoise_mc=True,
        noise_tr=2,
        noise_deint=DeintMethod.GENERATE,
        stabilize_noise=True,
    )
