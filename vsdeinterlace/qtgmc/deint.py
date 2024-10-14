__all__ = ["QTGMC"]

import vapoursynth as vs
from typing import Optional, Union, Any, Mapping

from stgpytools import CustomIntEnum
from vsdenoise import SearchMode

from vsdeinterlace.qtgmc.presets import (
    QTGMCPreset,
    QTGMCPresets,
    QTGMCNoisePreset,
    QTGMCNoisePresets,
)
from vsdeinterlace.qtgmc.enums import DeintMethod, DenoiseMethod, EdiMethod, InputType


class RepairSettings:
    repair0: int
    """Repair motion search clip (0=off): repair unwanted blur after temporal smooth TR0."""

    repair1: int
    """Repair initial output clip (0=off): repair unwanted blur after temporal smooth TR1."""

    repair2: int
    """Repair final output clip (0=off): unwanted blur after temporal smooth TR2 (will also repair TR1 blur if Rep1 not used)."""

    rep_chroma: bool
    """Whether the repair modes affect chroma"""


class InterpolateSettings:
    edi_mode: EdiMethod
    """Interpolation method"""

    nn_size: int
    """
    Area around each pixel used as predictor for NNEDI3.
    A larger area is slower with better quality, read the NNEDI3 docs to see the area choices.
    Note: area sizes are not in increasing order (i.e. increased value doesn't always mean increased quality).
    """

    num_neurons: int
    """Controls number of neurons in NNEDI3, larger = slower and better quality but improvements are small."""

    edi_qual: int
    """Quality setting for NNEDI3. Higher values for better quality - but improvements are marginal."""

    edi_max_dist: int
    """Spatial search distance for finding connecting edges in EEDI3."""

    chroma_edi: EdiMethod
    """Interpolation method used for chroma."""

    edi_clip: vs.VideoNode
    """Provide externally created interpolated clip rather than use one of the above modes."""

    nnedi3_args: Mapping[str, Any]
    """Additional arguments to pass to NNEDI3."""

    eedi3_args: Mapping[str, Any]
    """Additional arguments to pass to EEDI3."""


class SharpenSettings:
    sharpness: float
    """How much to resharpen the temporally blurred clip (default is always 1.0 unlike original TGMC)."""

    sharp_mode: int
    """
    Resharpening mode.
        0 = none
        1 = difference from 3x3 blur kernel
        2 = vertical max/min average + 3x3 kernel
    """
    sharp_limit_mode: int
    """
    Sharpness limiting.
        0 = off
        [1 = spatial, 2 = temporal]: before final temporal smooth
        [3 = spatial, 4 = temporal]: after final temporal smooth
    """

    sharp_limit_rad: int
    """
    Temporal or spatial radius used with sharpness limiting (depends on `sharp_limit_mode`).
    Temporal radius can only be 0, 1 or 3.
    """

    sharp_overshoot: int
    """
    Amount of overshoot allowed with temporal sharpness limiting (`sharp_limit_mode`=2,4),
    i.e. allow some oversharpening.
    """

    sharp_thin: float
    """
    How much to thin down 1-pixel wide lines that have been widened
    due to interpolation into neighboring field lines.
    """

    sharp_back_blend: int
    """
    Back blend (blurred) difference between pre & post sharpened clip (minor fidelity improvement).
        0 = off
        1 = before (1st) sharpness limiting
        2 = after (1st) sharpness limiting
        3 = both
    """


class MotionEstimationSettings:
    search_clip_pp: int
    """
    Pre-filtering for motion search clip.
        0 = none
        1 = simple blur
        2 = Gauss blur
        3 = Gauss blur + edge soften
    """

    sub_pel: int
    """
    Sub-pixel accuracy for motion analysis.
        1 = 1 pixel
        2 = 1/2 pixel
        4 = 1/4 pixel
    """

    sub_pel_interp: int
    """
    Interpolation used for sub-pixel motion analysis.
        0 = bilinear (soft)
        1 = bicubic (sharper)
        2 = Weiner (sharpest)
    """

    block_size: int
    """Size of blocks (square) that are matched during motion analysis."""

    overlap: int
    """
    How much to overlap motion analysis blocks (requires more blocks,
    but essential to smooth block edges in motion compensation).
    """

    search: SearchMode
    """Search method used for matching motion blocks."""

    search_param: int
    """Parameter for search method chosen. For hexagon search it is the search range."""

    pel_search: int
    """Search parameter (as above) for the finest sub-pixel level (see `sub_pel`)."""

    chroma_motion: bool
    """
    Whether to consider chroma when analyzing motion.
    Setting to false gives good speed-up, but may very occasionally make incorrect motion decision.
    """

    true_motion: bool
    """Whether to use the 'truemotion' defaults from MAnalyse (see MVTools2 documentation)."""

    coherence: int
    """
    Motion vector field coherence - how much the motion analysis
    favors similar motion vectors for neighboring blocks.
    Should be scaled by BlockSize*BlockSize/64.
    """

    coherence_sad: int
    """
    How much to reduce need for vector coherence (i.e. `coherence` above)
    if prediction of motion vector from neighbors is poor,
    typically in areas of complex motion.
    This value is scaled in MVTools (unlike `coherence`).
    """

    penalty_new: int
    """
    Penalty for choosing a new motion vector for a block over an existing one -
    avoids choosing new vectors for minor gain.
    """

    penalty_level: int
    """
    Mode for scaling lambda across different sub-pixel levels - see MVTools2 documentation for choices.
    """

    global_motion: bool
    """Whether to estimate camera motion to assist in selecting block motion vectors."""

    dct: int
    """
    Modes to use DCT (frequency analysis) or SATD as part of the block matching process -
    see MVTools2 documentation for choices.
    """

    th_sad1: int
    """
    SAD threshold for block match on shimmer-removing temporal smooth (TR1).
    Increase to reduce bob-shimmer more (may smear/blur).
    """

    th_sad2: int
    """
    SAD threshold for block match on final denoising temporal smooth (TR2).
    Increase to strengthen final smooth (may smear/blur).
    """

    th_scd1: int
    """Scene change detection parameter 1 - see MVTools documentation."""

    th_scd2: int
    """Scene change detection parameter 2 - see MVTools documentation."""

    fast_ma: bool
    """Use 8-bit for faster motion analysis when using high bit depth input."""

    extended_search: bool
    """Use wider search range for hex and umh search method."""

    refine_motion: bool
    """
    Refines and recalculates motion data of previously estimated motion vectors
    with new parameters set (e.g. lesser block size).
    The two-stage method may be also useful for more stable (robust) motion estimation.
    """


class SourceMatchSettings:
    source_match: int
    """
    0 = source-matching off (standard algorithm)
    1 = basic source-match
    2 = refined match
    3 = twice refined match
    """

    match_edi: EdiMethod
    """
    Override default interpolation method for basic source-match.
    Default method is same as main `edi_mode` setting (usually NNEDI3).
    Only need to override if using slow method for main interpolation (e.g. EEDI3)
    and want a faster method for source-match.
    """

    match_edi2: EdiMethod
    """
    Override interpolation method for refined source-match.
    Can be a good idea to pick match_edi2="Bob" for speed.
    """

    match_tr2: int
    """
    Temporal radius for refined source-matching.
    2=smoothness, 1=speed/sharper, 0=not recommended.
    Differences are very marginal.
    Basic source-match doesn't need this setting as its temporal radius must match TR1 core setting
    (i.e. there is no MatchTR1).
    """

    match_enhance: float
    """
    Enhance the detail found by source-match modes 2 & 3.
    A slight cheat - will enhance noise if set too strong. Best set < 1.0.
    """


class NoiseSettings:
    noise_process: int
    """
    0 = disable
    1 = denoise source & optionally restore some noise back at end of script [use for stronger denoising]
    2 = identify noise only & optionally restore some after QTGMC smoothing [for grain retention / light denoising]
    """

    ez_denoise: float
    """
    Automatic setting to denoise source. Set > 0.0 to enable.
    Higher values denoise more. Can use ShowNoise to help choose value.
    """

    ez_keep_grain: float
    """
    Automatic setting to retain source grain/detail. Set > 0.0 to enable.
    Higher values retain more grain. A good starting point = 1.0.
    """

    denoiser: DenoiseMethod
    """
    Select denoiser to use for noise bypass / denoising.
    """

    fft_threads: int
    """Number of threads to use if using "fft3dfilter" for Denoiser."""

    denoise_mc: bool
    """
    Whether to provide a motion-compensated clip to the denoiser for better noise vs detail detection
    (will be a little slower).
    """

    noise_tr: int
    """
    Temporal radius used when analyzing clip for noise extraction.
    Higher values better identify noise vs detail but are slower.
    """

    sigma: float
    """
    Amount of noise known to be in the source, sensible values vary by source and denoiser, so experiment.
    Use ShowNoise to help.
    """

    chroma_noise: bool
    """
    When processing noise (NoiseProcess > 0), whether to process chroma noise or not
    (luma noise is always processed).
    """

    show_noise: Union[bool, float]
    """
    Display extracted and "deinterlaced" noise rather than normal output.
    Set to true or false, or set a value (around 4 to 16) to specify
    contrast for displayed noise. Visualising noise helps to determine
    suitable value for Sigma or EZDenoise - want to see noise and noisy detail,
    but not too much clean structure or edges - fairly subjective.
    """

    grain_restore: float
    """
    How much removed noise/grain to restore before final temporal smooth.
    Retain "stable" grain and some detail (effect depends on TR2).
    """

    noise_restore: float
    """How much removed noise/grain to restore after final temporal smooth. Retains any kind of noise."""

    noise_deint: DeintMethod
    """
    When noise is taken from interlaced source, how to 'deinterlace' it before restoring.
    "Bob" & "DoubleWeave" are fast but with minor issues: "Bob" is coarse and "Doubleweave" lags by one frame.
    "Generate" is a high quality mode that generates fresh noise lines, but it is slower.
    """

    stabilize_noise: bool
    """
    Use motion compensation to limit shimmering and strengthen detail within the restored noise.
    Recommended for "Generate" mode.
    """


class ShutterSettings:
    shutter_blur: int
    """
    0=Off, 1=Enable, 2,3=Higher precisions (slower).
    Higher precisions reduce blur "bleeding" into static areas a little.
    """

    shutter_angle_src: float
    """
    Shutter angle used in source. If necessary, estimate from motion blur seen in a single frame.
    0=pin-sharp, 360=fully blurred from frame to frame.
    """

    shutter_angle_out: float
    """
    Shutter angle to simulate in output. Extreme values may be rejected (depends on other settings).
    Cannot reduce motion blur already in the source.
    """

    shutter_blur_limit: int
    """
    Limit motion blur where motion lower than given value.
    Increase to reduce blur "bleeding". 0=Off. Sensible range around 2-12.
    """


class QTGMC(CustomIntEnum):
    input_type: InputType
    """Type of input being processed by QTGMC"""

    tff: bool
    """True if source material is top-field first, False if bottom-field first"""

    tr0: int
    """
    Temporal binomial smoothing radius used to create motion search clip.
    In general: 2=quality, 1=speed, 0=don't use.
    """

    tr1: int
    """
    Temporal binomial smoothing radius used on interpolated clip for initial output.
    In general: 2=quality, 1=speed, 0=don't use.
    """

    tr2: int
    """
    Temporal linear smoothing radius used for final stabilization / denoising.
    Increase for smoother output.
    """

    lossless: int
    """
    Puts exact source fields into result & cleans any artefacts.
    0=off, 1=after final temporal smooth, 2=before resharpening.
    Adds some extra detail but:
    mode 1 gets shimmer / minor combing,
    mode 2 is more stable/tweakable but not exactly lossless.
    """

    prog_sad_mask: float
    """
    Only applies to progressive input types.
    If ProgSADMask > 0.0 then blend interlaced and progressive input modes based on block motion SAD.
    Higher values help recover more detail, but repair less artefacts.
    Reasonable range about 2.0 to 20.0, or 0.0 for no blending.
    """

    fps_divisor: int
    """
    1=Double-rate output, 2=Single-rate output.
    Higher values can be used too (e.g. 60fps & FPSDivisor=3 gives 20fps output).
    """

    border: bool
    """
    Pad a little vertically while processing (doesn't affect output size) -
    set true you see flickering on the very top or bottom line of the output.
    If you have wider edge effects than that, you should crop afterwards instead.
    """

    precise: bool
    """Set to false to use faster algorithms with *very* slight imprecision in places."""

    show_settings: bool
    """Display all the current parameter values - useful to find preset defaults."""

    force_tr: int
    """
    Ensure globally exposed motion vectors are calculated to this radius even if not needed by QTGMC.
    """

    strength: float
    """
    With this parameter you control the strength of the brightening of the prefilter clip for motion analysis.
    This is good when problems with dark areas arise.
    """

    amp: float
    """
    Use this together with Str (active when Str is different from 1.0).
    This defines the amplitude of the brightening in the luma range,
    for example by using 1.0 all the luma range will be used and the brightening
    will find its peak at luma value 128 in the original.
    """

    repair: RepairSettings
    interp: InterpolateSettings
    sharp: SharpenSettings
    me: MotionEstimationSettings
    source_match: SourceMatchSettings
    noise: NoiseSettings
    shutter: ShutterSettings

    def __init__(
        self,
        preset: QTGMCPreset = QTGMCPresets.SLOWER,
        match_preset: Optional[QTGMCPreset] = None,
        match_preset2: Optional[QTGMCPreset] = None,
        noise_preset: QTGMCNoisePreset = QTGMCNoisePresets.FAST,
        input_type: InputType = InputType.INTERLACED,
        tff: Optional[bool] = None,
        tr0: Optional[int] = None,
        tr1: Optional[int] = None,
        tr2: Optional[int] = None,
        repair0: Optional[int] = None,
        repair1: int = 0,
        repair2: Optional[int] = None,
        edi_mode: Optional[EdiMethod] = None,
        rep_chroma: bool = True,
        nn_size: Optional[int] = None,
        num_neurons: Optional[int] = None,
        edi_qual: int = 1,
        edi_max_dist: Optional[int] = None,
        chroma_edi: Optional[EdiMethod] = None,
        edi_clip: Optional[vs.VideoNode] = None,
        sharpness: Optional[float] = None,
        sharp_mode: Optional[int] = None,
        sharp_limit_mode: Optional[int] = None,
        sharp_limit_rad: Optional[int] = None,
        sharp_overshoot: int = 0,
        sharp_thin: float = 0.0,
        sharp_back_blend: Optional[int] = None,
        search_clip_pp: Optional[int] = None,
        sub_pel: Optional[int] = None,
        sub_pel_interp: int = 2,
        block_size: Optional[int] = None,
        overlap: Optional[int] = None,
        search: Optional[SearchMode] = None,
        search_param: Optional[int] = None,
        pel_search: Optional[int] = None,
        chroma_motion: Optional[bool] = None,
        true_motion: bool = False,
        coherence: Optional[int] = None,
        coherence_sad: Optional[int] = None,
        penalty_new: Optional[int] = None,
        penalty_level: Optional[int] = None,
        global_motion: bool = True,
        dct: int = 0,
        th_sad1: int = 640,
        th_sad2: int = 256,
        th_scd1: int = 180,
        th_scd2: int = 98,
        source_match: int = 0,
        match_edi: Optional[EdiMethod] = None,
        match_edi2: Optional[EdiMethod] = None,
        match_tr2: int = 1,
        match_enhance: float = 0.5,
        lossless: int = 0,
        noise_process: Optional[int] = None,
        ez_denoise: Optional[float] = None,
        ez_keep_grain: Optional[float] = None,
        denoiser: Optional[DenoiseMethod] = None,
        fft_threads: int = 1,
        denoise_mc: Optional[bool] = None,
        noise_tr: Optional[int] = None,
        sigma: Optional[float] = None,
        chroma_noise: bool = False,
        show_noise: Union[bool, float] = 0.0,
        grain_restore: Optional[float] = None,
        noise_restore: Optional[float] = None,
        noise_deint: Optional[DeintMethod] = None,
        stabilize_noise: Optional[bool] = None,
        prog_sad_mask: Optional[float] = None,
        fps_divisor: int = 1,
        shutter_blur: int = 0,
        shutter_angle_src: float = 180.0,
        shutter_angle_out: float = 180.0,
        shutter_blur_limit: int = 4,
        border: bool = False,
        precise: Optional[bool] = None,
        show_settings: bool = False,
        global_names: str = "QTGMC",
        prev_globals: str = "Replace",
        force_tr: int = 0,
        strength: float = 2.0,
        amp: float = 0.0625,
        fast_ma: bool = False,
        extended_search: bool = False,
        refine_motion: bool = False,
        nnedi3_args: Mapping[str, Any] = {},
        eedi3_args: Mapping[str, Any] = {},
    ) -> None:
        """
        Initializes the QTGMC processor.
        Any custom parameters specified will override the values from the preset.
        """
        pass
