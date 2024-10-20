__all__ = ["QTGMC"]

import math

import vapoursynth as vs
from typing import Optional, Union, Any, Mapping

from stgpytools import CustomIntEnum, fallback
from vsdenoise import SearchMode, prefilter_to_full_range, nl_means, BM3D
from vsexprtools import complexpr_available, norm_expr
from vsrgtools import gauss_blur
from vstools import (
    FieldBased,
    get_depth,
    scale_value,
    check_variable,
    depth,
    DitherType,
    check_ref_clip,
    normalize_planes,
    PlanesT,
)

from vsdeinterlace.qtgmc.presets import (
    QTGMCPreset,
    QTGMCPresets,
    QTGMCNoisePreset,
    QTGMCNoisePresets,
)
from vsdeinterlace.qtgmc.enums import DeintMethod, DenoiseMethod, EdiMethod, InputType

core = vs.core


class RepairSettings(dict[str, Any]):
    repair0: int
    repair1: int
    repair2: int
    rep_chroma: bool


class InterpolateSettings(dict[str, Any]):
    edi_mode: EdiMethod
    nn_size: int
    num_neurons: int
    edi_qual: int
    edi_max_dist: int
    chroma_edi: EdiMethod
    nnedi3_args: Mapping[str, Any]
    eedi3_args: Mapping[str, Any]


class SharpenSettings(dict[str, Any]):
    sharpness: float
    sharp_mode: int
    sharp_limit_mode: int
    sharp_limit_rad: int
    sharp_overshoot: int
    sharp_thin: float
    sharp_back_blend: int
    spatial_sharp_limit: bool
    temporal_sharp_limit: bool
    sharp_mul: float
    sharp_adj: float


class MotionEstimationSettings(dict[str, Any]):
    search_clip_pp: int
    sub_pel: int
    sub_pel_interp: int
    block_size: int
    overlap: int
    search: SearchMode
    search_param: int
    pel_search: int
    chroma_motion: bool
    true_motion: bool
    coherence: int
    coherence_sad: int
    penalty_new: int
    penalty_level: int
    global_motion: bool
    dct: int
    th_sad1: int
    th_sad2: int
    th_scd1: int
    th_scd2: int
    fast_ma: bool
    refine_motion: bool


class SourceMatchSettings(dict[str, Any]):
    source_match: int
    match_edi: EdiMethod
    match_nn_size: int
    match_num_neurons: int
    match_edi_max_dist: int
    match_edi_qual: int
    match_edi2: EdiMethod
    match_nn_size2: int
    match_num_neurons2: int
    match_edi_max_dist2: int
    match_edi_qual2: int
    match_tr1: int
    match_tr2: int
    match_enhance: float


class NoiseSettings(dict[str, Any]):
    noise_process: int
    ez_denoise: float
    ez_keep_grain: float
    denoiser: DenoiseMethod
    denoise_mc: bool
    noise_tr: int
    noise_td: int
    sigma: float
    chroma_noise: bool
    show_noise: Union[bool, float]
    grain_restore: float
    noise_restore: float
    total_restore: float
    noise_deint: DeintMethod
    stabilize_noise: bool
    fft_threads: int


class MotionBlurSettings(dict[str, Any]):
    shutter_blur: int
    shutter_angle_src: float
    shutter_angle_out: float
    shutter_blur_limit: int


class QTGMCGlobals(dict[str, Any]):
    search_clip: Optional[vs.VideoNode]
    search_super: Optional[vs.VideoNode]
    b_vec1: Optional[vs.VideoNode]
    f_vec1: Optional[vs.VideoNode]
    b_vec2: Optional[vs.VideoNode]
    f_vec2: Optional[vs.VideoNode]
    b_vec3: Optional[vs.VideoNode]
    f_vec3: Optional[vs.VideoNode]


class QTGMC:
    # These values are set when this class is initialized, based on the parameters supplied.
    # They may be edited by a user, but it is not advised, as that will override validation.
    input_type: InputType
    tr0: int
    tr1: int
    tr2: int
    max_tr: int
    lossless: int
    prog_sad_mask: float
    fps_divisor: int
    border: bool
    precise: bool
    strength: float
    amp: float
    repair: RepairSettings
    interp: InterpolateSettings
    sharp: SharpenSettings
    me: MotionEstimationSettings
    source_match: Optional[SourceMatchSettings]
    noise: NoiseSettings
    motion_blur: MotionBlurSettings

    # These values are set at runtime by `process`.
    # They may alternatively be provided by the user as parameters in `process`,
    # or accessed as outputs after running `process` via the `get_globals` function.
    # This replaces the use of globals that was present in `havsfunc.QTGMC`.
    _search_clip: Optional[vs.VideoNode]
    _search_super: Optional[vs.VideoNode]
    _b_vec1: Optional[vs.VideoNode]
    _f_vec1: Optional[vs.VideoNode]
    _b_vec2: Optional[vs.VideoNode]
    _f_vec2: Optional[vs.VideoNode]
    _b_vec3: Optional[vs.VideoNode]
    _f_vec3: Optional[vs.VideoNode]

    # These values are internal and are set at the start of each run of `process`
    # (or are constant) and shared between multiple functions.
    _matrix: list[int] = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    _tff: bool
    _is_gray: bool
    _bit_depth: int
    _neutral: int
    _sharp_overshoot: float
    _noise_center: float
    _cm_planes: list[int]
    _cn_planes: list[int]

    def __init__(
        self,
        preset: QTGMCPreset = QTGMCPresets.SLOWER,
        match_preset: Optional[QTGMCPreset] = None,
        match_preset2: Optional[QTGMCPreset] = None,
        noise_preset: QTGMCNoisePreset = QTGMCNoisePresets.FAST,
        input_type: InputType = InputType.INTERLACED,
        tr0: Optional[int] = None,
        tr1: Optional[int] = None,
        tr2: Optional[int] = None,
        repair0: Optional[int] = None,
        repair1: int = 0,
        repair2: Optional[int] = None,
        rep_chroma: bool = True,
        edi_mode: Optional[EdiMethod] = None,
        nn_size: Optional[int] = None,
        num_neurons: Optional[int] = None,
        edi_qual: int = 1,
        edi_max_dist: Optional[int] = None,
        chroma_edi: Optional[EdiMethod] = None,
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
        force_tr: int = 0,
        strength: float = 2.0,
        amp: float = 0.0625,
        fast_ma: bool = False,
        refine_motion: bool = False,
        nnedi3_args: Mapping[str, Any] = {},
        eedi3_args: Mapping[str, Any] = {},
    ) -> None:
        """
        Initializes the QTGMC processor.
        Any custom parameters specified will override the values from the preset.

        :param preset:              Speed/quality preset.
                                    Default is Slower.
        :param match_preset:        Speed/quality preset for basic source-match processing.
                                    Ideal choice is the same as main preset,
                                    but can choose a faster setting (but not a slower setting).
                                    Default is same as main `preset`.
        :param match_preset2:       Speed/quality preset for refined source-match processing.
                                    Default is 2 steps faster than `match_preset`.
                                    Faster settings are usually sufficient but can use slower settings
                                    if you get extra aliasing in this mode.
        :param noise_preset:        Speed/quality preset for noise processing
                                    Default is Fast.
        :param input_type:          Type of input being processed, whether interlaced or progressive.
        :param tr0:                 Temporal binomial smoothing radius used to create motion search clip.
                                    In general: 2=quality, 1=speed, 0=don't use.
        :param tr1:                 Temporal binomial smoothing radius used on interpolated clip for initial output.
                                    In general: 2=quality, 1=speed, 0=don't use.
        :param tr2:                 Temporal linear smoothing radius used for final stabilization / denoising.
                                    Increase for smoother output.
        :param repair0:             Repair motion search clip (0=off): repair unwanted blur after temporal smooth `tr0`.
        :param repair1:             Repair initial output clip (0=off): repair unwanted blur after temporal smooth `tr1`.
        :param repair2:             Repair final output clip (0=off): unwanted blur after temporal smooth `tr2`
                                    (will also repair `tr1` blur if `repair1` not used).
        :param rep_chroma:          Whether the repair modes affect chroma.
        :param edi_mode:            Interpolation method to use.
        :param nn_size:             Area around each pixel used as predictor for NNEDI3.
                                    A larger area is slower with better quality,
                                    read the NNEDI3 docs to see the area choices.
                                    Note: area sizes are not in increasing order
                                    (i.e. increased value doesn't always mean increased quality).
        :param num_neurons:         Controls number of neurons in NNEDI3,
                                    larger = slower and better quality but improvements are small.
        :param edi_qual:            Quality setting for NNEDI3. Higher values for better quality,
                                    but improvements are marginal.
        :param edi_max_dist:        Spatial search distance for finding connecting edges in EEDI3.
        :param chroma_edi:          Interpolation method used for chroma.
                                    If set to None, will use same mode as luma.
        :param sharpness:           How much to resharpen the temporally blurred clip.
        :param sharp_mode:          Resharpening mode.
                                    0 = none
                                    1 = difference from 3x3 blur kernel
                                    2 = vertical max/min average + 3x3 kernel
        :param sharp_limit_mode:     Sharpness limiting.
                                    0 = off
                                    [1 = spatial, 2 = temporal]: before final temporal smooth
                                    [3 = spatial, 4 = temporal]: after final temporal smooth
        :param sharp_limit_rad:     Temporal or spatial radius used with sharpness limiting (depends on `sharp_limit_mode`).
                                    Temporal radius can only be 0, 1 or 3.
        :param sharp_overshoot:     Amount of overshoot allowed with temporal sharpness limiting (`sharp_limit_mode`=2,4),
                                    i.e. allow some oversharpening.
        :param sharp_thin:          How much to thin down 1-pixel wide lines that have been widened
                                    due to interpolation into neighboring field lines.
        :param sharp_back_blend:     Back blend (blurred) difference between pre & post sharpened clip (minor fidelity improvement).
                                    0 = off
                                    1 = before (1st) sharpness limiting
                                    2 = after (1st) sharpness limiting
                                    3 = both
        :param search_clip_pp:      Pre-filtering for motion search clip.
                                    0 = none
                                    1 = simple blur
                                    2 = Gauss blur
                                    3 = Gauss blur + edge soften
        :param sub_pel:             Sub-pixel accuracy for motion analysis.
                                    1 = 1 pixel
                                    2 = 1/2 pixel
                                    4 = 1/4 pixel
        :param sub_pel_interp:      Interpolation used for sub-pixel motion analysis.
                                    0 = bilinear (soft)
                                    1 = bicubic (sharper)
                                    2 = Weiner (sharpest)
        :param block_size:          Size of blocks (square) that are matched during motion analysis.
        :param overlap:             How much to overlap motion analysis blocks (requires more blocks,
                                    but essential to smooth block edges in motion compensation).
        :param search:              Search method used for matching motion blocks.
        :param search_param:        Parameter for search method chosen. For hexagon search it is the search range.
        :param pel_search:          Search parameter (as above) for the finest sub-pixel level (see `sub_pel`).
        :param chroma_motion:       Whether to consider chroma when analyzing motion.
                                    Setting to False gives good speed-up,
                                    but may very occasionally make incorrect motion decision.
        :param true_motion:         Whether to use the 'truemotion' defaults from MAnalyse (see MVTools2 documentation).
        :param coherence:           Motion vector field coherence - how much the motion analysis
                                    favors similar motion vectors for neighboring blocks.
                                    Should be scaled by BlockSize*BlockSize/64.
        :param coherence_sad:       How much to reduce need for vector coherence (i.e. `coherence` above)
                                    if prediction of motion vector from neighbors is poor,
                                    typically in areas of complex motion.
                                    This value is scaled in MVTools (unlike `coherence`).
        :param penalty_new:         Penalty for choosing a new motion vector for a block over an existing one -
                                    avoids choosing new vectors for minor gain.
        :param penalty_level:       Mode for scaling lambda across different sub-pixel levels - see MVTools2 documentation for choices.
        :param global_motion:       Whether to estimate camera motion to assist in selecting block motion vectors.
        :param dct:                 Modes to use DCT (frequency analysis) or SATD as part of the block matching process -
                                    see MVTools2 documentation for choices.
        :param th_sad1:             SAD threshold for block match on shimmer-removing temporal smooth (TR1).
                                    Increase to reduce bob-shimmer more (may smear/blur).
        :param th_sad2:             SAD threshold for block match on final denoising temporal smooth (TR2).
                                    Increase to strengthen final smooth (may smear/blur).
        :param th_scd1:             Scene change detection parameter 1 - see MVTools documentation.
        :param th_scd2:             Scene change detection parameter 2 - see MVTools documentation.
        :param source_match:        Source matching mode.
                                    0 = source-matching off (standard algorithm)
                                    1 = basic source-match
                                    2 = refined match
                                    3 = twice refined match
        :param match_edi:           Override default interpolation method for basic source-match.
                                    Default method is same as main `edi_mode` setting (usually NNEDI3).
                                    Only need to override if using slow method for main interpolation (e.g. EEDI3)
                                    and want a faster method for source-match.
        :param match_edi2:          Override interpolation method for refined source-match.
                                    Can be a good idea to pick match_edi2="Bob" for speed.
        :param match_tr2:           Temporal radius for refined source-matching.
                                    2=smoothness, 1=speed/sharper, 0=not recommended.
                                    Differences are very marginal.
                                    Basic source-match doesn't need this setting as its temporal radius must match TR1 core setting
                                    (i.e. there is no MatchTR1).
        :param match_enhance:       Enhance the detail found by source-match modes 2 & 3.
                                    A slight cheat - will enhance noise if set too strong. Best set < 1.0.
        :param lossless:            Puts exact source fields into result & cleans any artefacts.
                                    0=off, 1=after final temporal smooth, 2=before resharpening.
                                    Adds some extra detail but:
                                    mode 1 gets shimmer / minor combing,
                                    mode 2 is more stable/tweakable but not exactly lossless.
        :param noise_process:       0 = disable
                                    1 = denoise source & optionally restore some noise back at end of script [use for stronger denoising]
                                    2 = identify noise only & optionally restore some after QTGMC smoothing [for grain retention / light denoising]
        :param ez_denoise:          Automatic setting to denoise source. Set > 0.0 to enable.
                                    Higher values denoise more. Can use ShowNoise to help choose value.
        :param ez_keep_grain:       Automatic setting to retain source grain/detail. Set > 0.0 to enable.
                                    Higher values retain more grain. A good starting point = 1.0.
        :param denoiser:            Select denoiser to use for noise bypass / denoising.
        :param fft_threads:         Number of threads to use if using "fft3dfilter" for Denoiser.
        :param denoise_mc:          Whether to provide a motion-compensated clip to the denoiser for better noise vs detail detection
                                    (will be a little slower).
        :param noise_tr:            Temporal radius used when analyzing clip for noise extraction.
                                    Higher values better identify noise vs detail but are slower.
        :param sigma:               Amount of noise known to be in the source, sensible values vary by source and denoiser, so experiment.
                                    Use ShowNoise to help.
        :param chroma_noise:        When processing noise (NoiseProcess > 0), whether to process chroma noise or not
                                    (luma noise is always processed).
        :param show_noise:          Display extracted and "deinterlaced" noise rather than normal output.
                                    Set to true or false, or set a value (around 4 to 16) to specify
                                    contrast for displayed noise. Visualising noise helps to determine
                                    suitable value for Sigma or EZDenoise - want to see noise and noisy detail,
                                    but not too much clean structure or edges - fairly subjective.
        :param grain_restore:       How much removed noise/grain to restore before final temporal smooth.
                                    Retain "stable" grain and some detail (effect depends on TR2).
        :param noise_restore:       How much removed noise/grain to restore after final temporal smooth. Retains any kind of noise.
        :param noise_deint:         When noise is taken from interlaced source, how to 'deinterlace' it before restoring.
                                    "Bob" & "DoubleWeave" are fast but with minor issues: "Bob" is coarse and "Doubleweave" lags by one frame.
                                    "Generate" is a high quality mode that generates fresh noise lines, but it is slower.
        :param stabilize_noise:     Use motion compensation to limit shimmering and strengthen detail within the restored noise.
                                    Recommended for "Generate" mode.
        :param prog_sad_mask:       Only applies to progressive input types.
                                    If ProgSADMask > 0.0 then blend interlaced and progressive input modes based on block motion SAD.
                                    Higher values help recover more detail, but repair less artefacts.
                                    Reasonable range about 2.0 to 20.0, or 0.0 for no blending.
        :param fps_divisor:         1=Double-rate output, 2=Single-rate output.
                                    Higher values can be used too (e.g. 60fps & FPSDivisor=3 gives 20fps output).
        :param shutter_blur:        0=Off, 1=Enable, 2,3=Higher precisions (slower).
                                    Higher precisions reduce blur "bleeding" into static areas a little.
        :param shutter_angle_src:   Shutter angle used in source. If necessary, estimate from motion blur seen in a single frame.
                                    0=pin-sharp, 360=fully blurred from frame to frame.
        :param shutter_angle_out:   Shutter angle to simulate in output. Extreme values may be rejected (depends on other settings).
                                    Cannot reduce motion blur already in the source.
        :param shutter_blur_limit:  Limit motion blur where motion lower than given value.
                                    Increase to reduce blur "bleeding". 0=Off. Sensible range around 2-12.
        :param border:              Pad a little vertically while processing (doesn't affect output size) -
                                    set true you see flickering on the very top or bottom line of the output.
                                    If you have wider edge effects than that, you should crop afterwards instead.
        :param precise:             Set to false to use faster algorithms with *very* slight imprecision in places.
        :param force_tr:            Ensure globally exposed motion vectors are calculated to this radius even if not needed by QTGMC.
        :param strength:            With this parameter you control the strength of the brightening of the prefilter clip for motion analysis.
                                    This is good when problems with dark areas arise.
        :param amp:                 Use this together with Str (active when Str is different from 1.0).
                                    This defines the amplitude of the brightening in the luma range,
                                    for example by using 1.0 all the luma range will be used and the brightening
                                    will find its peak at luma value 128 in the original.
        :param fast_ma:             Use 8-bit for faster motion analysis when using high bit depth input.
        :param refine_motion:       Refines and recalculates motion data of previously estimated motion vectors
                                    with new parameters set (e.g. lesser block size).
                                    The two-stage method may be also useful for more stable (robust) motion estimation.
        :param nnedi3_args:         Additional arguments to pass to NNEDI3.
        :param eedi3_args:          Additional arguments to pass to EEDI3.
        """

        self.input_type = input_type
        self.tr0 = fallback(tr0, preset.tr0)
        self.tr1 = fallback(tr1, preset.tr1)
        self.tr2 = fallback(tr2, max(preset.tr2, 1) if source_match > 0 else preset.tr2)
        self.precise = fallback(precise, preset.precise)
        self.prog_sad_mask = (
            fallback(prog_sad_mask, preset.prog_sad_mask)
            if self.input_type == InputType.PROGRESSIVE_WITH_COMBING
            else 0.0
        )

        self.repair = RepairSettings(
            repair0=fallback(repair0, preset.repair0),
            repair2=fallback(repair2, preset.repair2),
            repair1=repair1,
            rep_chroma=rep_chroma,
        )

        self.interp = InterpolateSettings(
            edi_mode=fallback(edi_mode, preset.edi_mode),
            nn_size=fallback(nn_size, preset.nn_size),
            num_neurons=fallback(num_neurons, preset.num_neurons),
            edi_max_dist=fallback(edi_max_dist, preset.edi_max_dist),
            edi_qual=edi_qual,
            chroma_edi=chroma_edi,
            nnedi3_args=nnedi3_args,
            eedi3_args=eedi3_args,
        )

        # Sharpness defaults. Sharpness default is always 1.0 (0.2 with source-match),
        # but adjusted to give roughly same sharpness for all settings
        sharp_limit_rad = fallback(sharp_limit_rad, preset.sharp_limit_rad)
        sharp_mode = (
            0
            if sharpness is not None and sharpness <= 0
            else fallback(sharp_mode, preset.sharp_mode)
        )
        sharp_limit_mode = (
            0
            if sharp_limit_rad <= 0
            else fallback(
                sharp_limit_mode, 0 if source_match > 0 else preset.sharp_limit_mode
            )
        )
        sharpness = fallback(
            sharpness, 0.0 if sharp_mode <= 0 else 0.2 if source_match > 0 else 1.0
        )
        spatial_sharp_limit = (sharp_limit_mode in [1, 3],)
        temporal_sharp_limit = (sharp_limit_mode in [2, 4],)
        sharp_mul = 2 if temporal_sharp_limit else 1.5 if spatial_sharp_limit else 1
        self.sharp = SharpenSettings(
            sharp_mode=sharp_mode,
            sharp_limit_rad=sharp_limit_rad,
            sharp_back_blend=(
                0
                if sharp_mode <= 0
                else fallback(sharp_back_blend, preset.sharp_back_blend)
            ),
            sharp_limit_mode=sharp_limit_mode,
            sharpness=sharpness,
            sharp_overshoot=sharp_overshoot,
            sharp_thin=sharp_thin,
            spatial_sharp_limit=spatial_sharp_limit,
            temporal_sharp_limit=temporal_sharp_limit,
            sharp_mul=sharp_mul,
            sharp_adj=sharpness
            * (
                sharp_mul * (0.2 + self.tr1 * 0.15 + self.tr2 * 0.25)
                + (0.1 if sharp_mode == 1 else 0)
            ),
        )

        block_size = fallback(block_size, preset.block_size)
        self.me = MotionEstimationSettings(
            search_clip_pp=fallback(search_clip_pp, preset.search_clip_pp),
            sub_pel=fallback(sub_pel, preset.sub_pel),
            sub_pel_interp=sub_pel_interp,
            block_size=block_size,
            overlap=fallback(overlap, preset.overlap),
            search=fallback(search, preset.search),
            search_param=fallback(search_param, preset.search_param),
            pel_search=fallback(pel_search, preset.pel_search),
            chroma_motion=fallback(chroma_motion, preset.chroma_motion),
            true_motion=true_motion,
            coherence=fallback(
                coherence,
                (1000 if true_motion else 100) * block_size * block_size // 64,
            ),
            coherence_sad=fallback(coherence_sad, 1200 if true_motion else 400),
            penalty_new=fallback(penalty_new, 50 if true_motion else 25),
            penalty_level=fallback(penalty_level, 1 if true_motion else 0),
            global_motion=global_motion,
            dct=dct,
            th_sad1=th_sad1,
            th_sad2=th_sad2,
            th_scd1=th_scd1,
            th_scd2=th_scd2,
            fast_ma=fast_ma,
            refine_motion=refine_motion,
        )

        if (
            ez_denoise is not None
            and ez_denoise > 0
            and ez_keep_grain is not None
            and ez_keep_grain > 0
        ):
            raise vs.Error(
                "QTGMC: ez_denoise and ez_keep_grain cannot be used together"
            )
        noise_tr = fallback(noise_tr, noise_preset.noise_tr)
        stabilize_noise = fallback(stabilize_noise, noise_preset.stabilize_noise)
        if noise_process is None:
            if ez_denoise is not None and ez_denoise > 0:
                noise_process = 1
            elif (ez_keep_grain is not None and ez_keep_grain > 0) or preset.preset in [
                QTGMCPresets.PLACEBO.preset,
                QTGMCPresets.VERY_SLOW.preset,
            ]:
                noise_process = 2
            else:
                noise_process = 0
        if grain_restore is None:
            if ez_denoise is not None and ez_denoise > 0:
                grain_restore = 0.0
            elif ez_keep_grain is not None and ez_keep_grain > 0:
                grain_restore = 0.3 * math.sqrt(ez_keep_grain)
            else:
                grain_restore = [0.0, 0.7, 0.3][noise_process]
        if noise_restore is None:
            if ez_denoise is not None and ez_denoise > 0:
                noise_restore = 0.0
            elif ez_keep_grain is not None and ez_keep_grain > 0:
                noise_restore = 0.1 * math.sqrt(ez_keep_grain)
            else:
                noise_restore = [0.0, 0.3, 0.1][noise_process]
        if sigma is None:
            if ez_denoise is not None and ez_denoise > 0:
                sigma = ez_denoise
            elif ez_keep_grain is not None and ez_keep_grain > 0:
                sigma = 4.0 * ez_keep_grain
            else:
                sigma = 2.0
        if isinstance(show_noise, bool):
            show_noise = 10.0 if show_noise else 0.0
        if show_noise > 0:
            noise_process = 2
            noise_restore = 1.0
        if noise_process <= 0:
            noise_tr = 0
            grain_restore = 0.0
            noise_restore = 0.0
        total_restore = grain_restore + noise_restore
        if total_restore <= 0:
            stabilize_noise = False
        self.noise = NoiseSettings(
            noise_process=noise_process,
            ez_denoise=ez_denoise,
            ez_keep_grain=ez_keep_grain,
            denoiser=fallback(denoiser, noise_preset.denoiser),
            denoise_mc=fallback(denoise_mc, noise_preset.denoise_mc),
            noise_tr=noise_tr,
            noise_td=[1, 3, 5][noise_tr],
            noise_deint=fallback(noise_deint, noise_preset.noise_deint),
            stabilize_noise=stabilize_noise,
            sigma=sigma,
            chroma_noise=chroma_noise,
            show_noise=show_noise,
            grain_restore=grain_restore,
            noise_restore=noise_restore,
            total_restore=total_restore,
            fft_threads=fft_threads,
        )

        self.motion_blur = MotionBlurSettings(
            # Disable if motion blur output is same as input
            shutter_blur=(
                0
                if shutter_angle_out * fps_divisor == shutter_angle_src
                else shutter_blur
            ),
            shutter_angle_src=shutter_angle_src,
            shutter_angle_out=shutter_angle_out,
            shutter_blur_limit=shutter_blur_limit,
        )

        if match_preset is None or match_preset.preset > QTGMCPresets.ULTRA_FAST.preset:
            match_preset = QTGMCPreset.from_int(
                min(preset.preset, QTGMCPresets.ULTRA_FAST.preset)
            )
        if match_preset2 is None:
            match_preset2 = QTGMCPreset.from_int(
                min(match_preset.preset + 2, QTGMCPresets.ULTRA_FAST.preset)
            )
        if source_match > 0 and match_preset.preset < preset.preset:
            raise vs.Error(
                "QTGMC: Match Preset cannot use a slower setting than Preset"
            )

        self.source_match = SourceMatchSettings(
            source_match=source_match,
            match_edi=edi_mode,
            match_nn_size=self.interp.nn_size,
            match_num_neurons=self.interp.num_neurons,
            match_edi_max_dist=self.interp.edi_max_dist,
            match_edi_qual=self.interp.edi_qual,
            match_edi2=fallback(match_edi2, match_preset.edi_mode),
            match_nn_size2=match_preset.nn_size,
            match_num_neurons2=match_preset.num_neurons,
            match_edi_max_dist2=match_preset.edi_max_dist,
            match_edi_qual2=1,
            match_tr1=self.tr1,
            match_tr2=match_tr2,
            match_enhance=match_enhance,
        )
        if source_match > 0:
            # Basic source-match presets
            self.interp.nn_size = match_preset.nn_size
            self.interp.num_neurons = match_preset.num_neurons
            self.interp.edi_max_dist = match_preset.edi_max_dist
            self.interp.edi_qual = 1
            if match_edi is not None:
                self.interp.edi_mode = match_edi
            elif match_preset.preset == QTGMCPresets.ULTRA_FAST.preset:
                # Force bwdif for Ultra Fast basic source match
                self.interp.edi_mode = EdiMethod.BWDIF

        self.max_tr = max(
            self.sharp.sharp_limit_rad if self.sharp.temporal_sharp_limit else 0,
            self.source_match.match_tr2,
            self.tr1,
            self.tr2,
            self.noise.noise_tr,
        )
        if (
            self.prog_sad_mask > 0
            or self.noise.stabilize_noise
            or self.motion_blur.shutter_blur > 0
        ) and self.max_tr < 1:
            self.max_tr = 1
        self.max_tr = max(force_tr, self.max_tr)

        self.lossless = lossless
        self.border = border
        self.strength = strength
        self.amp = amp

    def process(
        self,
        clip: vs.VideoNode,
        tff: Optional[bool] = None,
        edi_clip: Optional[vs.VideoNode] = None,
        search_clip: Optional[vs.VideoNode] = None,
        search_super: Optional[vs.VideoNode] = None,
        b_vec1: Optional[vs.VideoNode] = None,
        f_vec1: Optional[vs.VideoNode] = None,
        b_vec2: Optional[vs.VideoNode] = None,
        f_vec2: Optional[vs.VideoNode] = None,
        b_vec3: Optional[vs.VideoNode] = None,
        f_vec3: Optional[vs.VideoNode] = None,
    ) -> vs.VideoNode:
        """
        Process a given `clip` using QTGMC to deinterlace and/or fix shimmering and combing.

        :param clip:         Clip to process
        :param tff:          True if source material is top-field first, False if bottom-field first.
                             None will attempt to infer from source clip.
        :param edi_clip:     Externally created interpolated clip to use rather than QTGMC's interpolation.
        :param search_clip:  Externally created filtered clip used for motion analysis.
        :param search_super: Externally created MVTools "super" clip for filtered clip.
        :param b_vec1:       Externally created backward motion vectors (temporal radius 1)
        :param f_vec1:       Externally created forward motion vectors (temporal radius 1)
        :param b_vec2:       Externally created backward motion vectors (temporal radius 2)
        :param f_vec2:       Externally created forward motion vectors (temporal radius 2)
        :param b_vec3:       Externally created backward motion vectors (temporal radius 3)
        :param f_vec3:       Externally created forward motion vectors (temporal radius 3)
        """

        if not isinstance(clip, vs.VideoNode):
            raise vs.Error("QTGMC: input is not a clip")

        if edi_clip is not None and not isinstance(edi_clip, vs.VideoNode):
            raise vs.Error("QTGMC: edi_clip is not a clip")

        if self.input_type != InputType.PROGRESSIVE and tff is None:
            tff = FieldBased.from_video(clip, True).is_tff

        self._tff = tff
        self._is_gray = clip.format.color_family == vs.GRAY
        self._bit_depth = get_depth(clip)
        self._neutral = 1 << (self._bit_depth - 1)
        self._sharp_overshoot = scale_value(
            self.sharp.sharp_overshoot, 8, self._bit_depth
        )
        self._noise_center = scale_value(128.5, 8, self._bit_depth)

        width = clip.width
        height = clip.height

        # Pad vertically during processing (to prevent artefacts at top & bottom edges)
        if self.border:
            height += 8
            clip = clip.resize.Point(width, height, src_top=-4, src_height=height)

        hpad = vpad = self.me.block_size

        # --- Motion Analysis

        # Bob the input as a starting point for the motion search clip
        if self.input_type == InputType.INTERLACED:
            bobbed = clip.resize.Bob(tff=self._tff, filter_param_a=0, filter_param_b=0.5)
        elif self.input_type == InputType.PROGRESSIVE:
            bobbed = clip
        else:
            bobbed = clip.std.Convolution(matrix=[1, 2, 1], mode="v")

        self._cm_planes = (
            [0, 1, 2] if self.me.chroma_motion and not self._is_gray else [0]
        )

        if isinstance(search_clip, vs.VideoNode):
            self._search_clip = search_clip
        else:
            self._search_clip = self._build_search_clip(bobbed, width, height)

        super_args = dict(pel=self.me.sub_pel, hpad=hpad, vpad=vpad)
        analyse_args = dict(
            blksize=self.me.block_size,
            overlap=self.me.overlap,
            search=self.me.search,
            searchparam=self.me.search_param,
            pelsearch=self.me.pel_search,
            truemotion=self.me.true_motion,
            lambda_=self.me.coherence,
            lsad=self.me.coherence_sad,
            pnew=self.me.penalty_new,
            plevel=self.me.penalty_level,
            global_=self.me.global_motion,
            dct=self.me.dct,
            chroma=self.me.chroma_motion,
        )
        recalculate_args = dict(
            thsad=self.me.th_sad1 // 2,
            blksize=max(self.me.block_size // 2, 4),
            search=self.me.search,
            searchparam=self.me.search_param,
            chroma=self.me.chroma_motion,
            truemotion=self.me.true_motion,
            pnew=self.me.penalty_new,
            overlap=max(self.me.overlap // 2, 2),
            dct=self.me.dct,
        )

        if self.max_tr > 0:
            if isinstance(search_super, vs.VideoNode):
                self._search_super = search_super
            else:
                self._search_super = self._search_clip.mv.Super(
                    sharp=self.me.sub_pel_interp,
                    chroma=self.me.chroma_motion,
                    **super_args,
                )
            if isinstance(b_vec1, vs.VideoNode):
                self._b_vec1 = b_vec1
            else:
                self._b_vec1 = self._search_super.mv.Analyse(
                    isb=True, delta=1, **analyse_args
                )
                if self.me.refine_motion:
                    self._b_vec1 = core.mv.Recalculate(
                        self._search_super, self._b_vec1, **recalculate_args
                    )
            if isinstance(f_vec1, vs.VideoNode):
                self._f_vec1 = f_vec1
            else:
                self._f_vec1 = self._search_super.mv.Analyse(
                    isb=False, delta=1, **analyse_args
                )
                if self.me.refine_motion:
                    self._f_vec1 = core.mv.Recalculate(
                        self._search_super, self._f_vec1, **recalculate_args
                    )
        if self.max_tr > 1:
            if isinstance(b_vec2, vs.VideoNode):
                self._b_vec2 = b_vec2
            else:
                self._b_vec2 = self._search_super.mv.Analyse(
                    isb=True, delta=2, **analyse_args
                )
                if self.me.refine_motion:
                    self._b_vec2 = core.mv.Recalculate(
                        self._search_super, self._b_vec2, **recalculate_args
                    )
            if isinstance(f_vec2, vs.VideoNode):
                self._f_vec2 = f_vec2
            else:
                self._f_vec2 = self._search_super.mv.Analyse(
                    isb=False, delta=2, **analyse_args
                )
                if self.me.refine_motion:
                    self._f_vec2 = core.mv.Recalculate(
                        self._search_super, self._f_vec2, **recalculate_args
                    )
        if self.max_tr > 2:
            if isinstance(b_vec3, vs.VideoNode):
                self._b_vec3 = b_vec3
            else:
                self._b_vec3 = self._search_super.mv.Analyse(
                    isb=True, delta=3, **analyse_args
                )
                if self.me.refine_motion:
                    self._b_vec3 = core.mv.Recalculate(
                        self._search_super, self._b_vec3, **recalculate_args
                    )
            if isinstance(f_vec3, vs.VideoNode):
                self._f_vec3 = f_vec3
            else:
                self._f_vec3 = self._search_super.mv.Analyse(
                    isb=False, delta=3, **analyse_args
                )
                if self.me.refine_motion:
                    self._f_vec3 = core.mv.Recalculate(
                        self._search_super, self._f_vec3, **recalculate_args
                    )

        # --- Noise Processing

        # Expand fields to full frame size before extracting noise
        # (allows use of motion vectors which are frame-sized)
        if self.noise.noise_process > 0:
            if self.input_type == InputType.INTERLACED:
                full_clip = clip.resize.Bob(tff=self._tff, filter_param_a=0, filter_param_b=1)
            else:
                full_clip = clip

            if self.noise.noise_tr > 0:
                full_super = full_clip.mv.Super(
                    levels=1, chroma=self.noise.chroma_noise, **super_args
                )

            self._cn_planes = (
                [0, 1, 2] if self.noise.chroma_noise and not self._is_gray else [0]
            )

            # Create a motion compensated temporal window around current frame and use to guide denoisers
            if not self.noise.denoise_mc or self.noise.noise_tr <= 0:
                noise_window = full_clip
            elif self.noise.noise_tr == 1:
                noise_window = core.std.Interleave(
                    [
                        core.mv.Compensate(
                            full_clip,
                            full_super,
                            self._f_vec1,
                            thscd1=self.me.th_scd1,
                            thscd2=self.me.th_scd2,
                        ),
                        full_clip,
                        core.mv.Compensate(
                            full_clip,
                            full_super,
                            self._b_vec1,
                            thscd1=self.me.th_scd1,
                            thscd2=self.me.th_scd2,
                        ),
                    ]
                )
            else:
                noise_window = core.std.Interleave(
                    [
                        core.mv.Compensate(
                            full_clip,
                            full_super,
                            self._f_vec2,
                            thscd1=self.me.th_scd1,
                            thscd2=self.me.th_scd2,
                        ),
                        core.mv.Compensate(
                            full_clip,
                            full_super,
                            self._f_vec1,
                            thscd1=self.me.th_scd1,
                            thscd2=self.me.th_scd2,
                        ),
                        full_clip,
                        core.mv.Compensate(
                            full_clip,
                            full_super,
                            self._b_vec1,
                            thscd1=self.me.th_scd1,
                            thscd2=self.me.th_scd2,
                        ),
                        core.mv.Compensate(
                            full_clip,
                            full_super,
                            self._b_vec2,
                            thscd1=self.me.th_scd1,
                            thscd2=self.me.th_scd2,
                        ),
                    ]
                )

            if self.noise.denoiser == DenoiseMethod.BM3D:
                dn_window = BM3D.denoise(
                    noise_window,
                    self.noise.sigma,
                    self.noise.noise_tr,
                    planes=self._cn_planes,
                )
            elif self.noise.denoiser == DenoiseMethod.DFTTEST:
                dn_window = noise_window.dfttest.DFTTest(
                    sigma=self.noise.sigma * 4,
                    tbsize=self.noise.noise_td,
                    planes=self._cn_planes,
                )
            elif self.noise.denoiser == DenoiseMethod.KNLMeans:
                dn_window = nl_means(
                    noise_window,
                    strength=self.noise.sigma,
                    tr=self.noise.noise_tr,
                    planes=self._cn_planes,
                )
            else:
                dn_window = noise_window.fft3dfilter.FFT3DFilter(
                    sigma=self.noise.sigma,
                    planes=self._cn_planes,
                    bt=self.noise.noise_td,
                    ncpu=self.noise.fft_threads,
                )

            # Rework denoised clip to match source format - various code paths here:
            # discard the motion compensation window, discard doubled lines (from point resize)
            # Also reweave to get interlaced noise if source was interlaced
            # (could keep the full frame of noise, but it will be poor quality from the point resize)
            if not self.noise.denoise_mc:
                if self.input_type != InputType.INTERLACED:
                    denoised = dn_window
                else:
                    denoised = (
                        dn_window.std.SeparateFields(tff=self._tff)
                        .std.SelectEvery(cycle=4, offsets=[0, 3])
                        .std.DoubleWeave(self._tff)[::2]
                    )
            elif self.input_type != InputType.INTERLACED:
                if self.noise.noise_tr <= 0:
                    denoised = dn_window
                else:
                    denoised = dn_window.std.SelectEvery(
                        cycle=self.noise.noise_td, offsets=self.noise.noise_tr
                    )
            else:
                denoised = (
                    dn_window.std.SeparateFields(tff=self._tff)
                    .std.SelectEvery(
                        cycle=self.noise.noise_td * 4,
                        offsets=[
                            self.noise.noise_tr * 2,
                            self.noise.noise_tr * 6 + 3,
                        ],
                    )
                    .std.DoubleWeave(self._tff)[::2]
                )

            if self.noise.total_restore > 0:
                # Get actual noise from difference.
                # Then 'deinterlace' where we have weaved noise -
                # create the missing lines of noise in various ways
                noise = core.std.MakeDiff(clip, denoised, planes=self._cn_planes)
                if self.input_type != InputType.INTERLACED:
                    deint_noise = noise
                elif self.noise.noise_deint == DeintMethod.BOB:
                    deint_noise = noise.resize.Bob(
                        tff=self._tff, filter_param_a=0, filter_param_b=0.5
                    )
                elif self.noise.noise_deint == DeintMethod.GENERATE:
                    deint_noise = self._generate_2nd_field_noise(noise, denoised)
                else:
                    deint_noise = noise.std.SeparateFields(tff=self._tff).std.DoubleWeave(
                        tff=self._tff
                    )

                # Motion-compensated stabilization of generated noise
                if self.noise.stabilize_noise:
                    noise_super = deint_noise.mv.Super(
                        sharp=self.me.sub_pel_interp,
                        levels=1,
                        chroma=self.noise.chroma_noise,
                        **super_args,
                    )
                    mc_noise = core.mv.Compensate(
                        deint_noise,
                        noise_super,
                        self._b_vec1,
                        thscd1=self.me.th_scd1,
                        thscd2=self.me.th_scd2,
                    )
                    expr = f"x {self._neutral} - abs y {self._neutral} - abs > x y ? 0.6 * x y + 0.2 * +"
                    final_noise = core.std.Expr(
                        [deint_noise, mc_noise],
                        expr=(
                            expr
                            if self.noise.chroma_noise or self._is_gray
                            else [expr, ""]
                        ),
                    )
                else:
                    final_noise = deint_noise

        # If NoiseProcess=1 denoise input clip.
        # If NoiseProcess=2 leave noise in the clip and let the temporal blurs
        # "denoise" it for a stronger effect
        inner_clip = denoised if self.noise.noise_process == 1 else clip

        # --- Interpolation

        # Support badly deinterlaced progressive content -
        # drop half the fields and reweave to get 1/2fps interlaced stream
        # appropriate for QTGMC processing
        if self.input_type == InputType.PROGRESSIVE_WITH_COMBING:
            edi_input = (
                inner_clip.std.SeparateFields(tff=self._tff)
                .std.SelectEvery(cycle=4, offsets=[0, 3])
                .std.DoubleWeave(self._tff)[::2]
            )
        else:
            edi_input = inner_clip

        # Create interpolated image as starting point for output
        if edi_clip is not None:
            edi1 = edi_clip.resize.Point(
                width,
                height,
                src_top=(edi_clip.height - height) // 2,
                src_height=height,
            )
        else:
            edi1 = self._interpolate(
                edi_input,
                bobbed,
                self.interp.edi_mode,
                self.interp.nn_size,
                self.interp.num_neurons,
                self.interp.edi_qual,
                self.interp.edi_max_dist,
            )

        # InputType=2,3: use motion mask to blend luma
        # between original clip & reweaved clip based on ProgSADMask setting.
        # Use chroma from original clip in any case
        if self.input_type != InputType.PROGRESSIVE_WITH_COMBING:
            edi = edi1
        elif self.prog_sad_mask <= 0:
            if not self._is_gray:
                edi = core.std.ShufflePlanes(
                    [edi1, inner_clip],
                    planes=[0, 1, 2],
                    colorfamily=clip.format.color_family,
                )
            else:
                edi = edi1
        else:
            input_type_blend = core.mv.Mask(
                depth(self._search_clip, 8, dither_type=DitherType.NONE),
                self._b_vec1,
                kind=1,
                ml=self.prog_sad_mask,
            )
            input_type_blend = depth(
                input_type_blend,
                self._search_clip.format.bits_per_sample,
                dither_type=DitherType.NONE,
            )
            edi = core.std.MaskedMerge(inner_clip, edi1, input_type_blend, planes=0)

        # Get the max/min value for each pixel over neighboring motion-compensated frames -
        # used for temporal sharpness limiting
        if self.tr1 > 0 or self.sharp.temporal_sharp_limit:
            edi_super = edi.mv.Super(
                sharp=self.me.sub_pel_interp, levels=1, **super_args
            )

        if self.sharp.temporal_sharp_limit:
            b_comp1 = core.mv.Compensate(
                edi,
                edi_super,
                self._b_vec1,
                thscd1=self.me.th_scd1,
                thscd2=self.me.th_scd2,
            )
            f_comp1 = core.mv.Compensate(
                edi,
                edi_super,
                self._f_vec1,
                thscd1=self.me.th_scd1,
                thscd2=self.me.th_scd2,
            )
            t_max = core.std.Expr(
                [core.std.Expr([edi, f_comp1], expr="x y max"), b_comp1],
                expr="x y max",
            )
            t_min = core.std.Expr(
                [core.std.Expr([edi, f_comp1], expr="x y min"), b_comp1],
                expr="x y min",
            )
            if self.sharp.sharp_limit_rad > 1:
                b_comp3 = core.mv.Compensate(
                    edi,
                    edi_super,
                    self._b_vec3,
                    thscd1=self.me.th_scd1,
                    thscd2=self.me.th_scd2,
                )
                f_comp3 = core.mv.Compensate(
                    edi,
                    edi_super,
                    self._f_vec3,
                    thscd1=self.me.th_scd1,
                    thscd2=self.me.th_scd2,
                )
                t_max = core.std.Expr(
                    [core.std.Expr([t_max, f_comp3], expr="x y max"), b_comp3],
                    expr="x y max",
                )
                t_min = core.std.Expr(
                    [core.std.Expr([t_min, f_comp3], expr="x y min"), b_comp3],
                    expr="x y min",
                )

        # --- Create basic output

        # Use motion vectors to blur interpolated image (edi) with motion-compensated previous and next frames.
        # As above, this is done to remove shimmer from alternate frames so the same binomial kernels are used.
        # However, by using motion-compensated smoothing this time we avoid motion blur.
        # The use of MDegrain1 (motion compensated) rather than TemporalSmooth makes
        # the weightings *look* different, but they evaluate to the same values

        # Create linear weightings of neighbors first
        # -2    -1    0     1     2
        if self.tr1 > 0:
            # 0.00  0.33  0.33  0.33  0.00
            degrain1 = core.mv.Degrain1(
                edi,
                edi_super,
                self._b_vec1,
                self._f_vec1,
                thsad=self.me.th_sad1,
                thscd1=self.me.th_scd1,
                thscd2=self.me.th_scd2,
            )
        if self.tr1 > 1:
            # 0.33  0.00  0.33  0.00  0.33
            degrain2 = core.mv.Degrain1(
                edi,
                edi_super,
                self._b_vec2,
                self._f_vec2,
                thsad=self.me.th_sad1,
                thscd1=self.me.th_scd1,
                thscd2=self.me.th_scd2,
            )

        # Combine linear weightings to give binomial weightings - TR1=0: (1), TR1=1: (1:2:1), TR1=2: (1:4:6:4:1)
        if self.tr1 <= 0:
            binomial1 = edi
        elif self.tr1 == 1:
            binomial1 = core.std.Merge(degrain1, edi, weight=0.25)
        else:
            binomial1 = core.std.Merge(
                core.std.Merge(degrain1, degrain2, weight=0.2), edi, weight=0.0625
            )

        # Remove areas of difference between smoothed image and interpolated image
        # that are not bob-shimmer fixes: repairs residual motion blur from temporal smooth
        if self.repair.repair1 <= 0:
            repair1 = binomial1
        else:
            repair1 = self._keep_only_bob_shimmer_fixes(
                binomial1, edi, self.repair.repair1, self.repair.rep_chroma
            )

        # Apply source match - use difference between output and source to successively refine output
        # [extracted to function to clarify main code path]
        if self.source_match.source_match <= 0:
            match = repair1
        else:
            match = self._apply_source_match(
                repair1,
                edi_clip,
                hpad,
                vpad,
            )

        # Lossless=2 - after preparing an interpolated, de-shimmered clip,
        # restore the original source fields into it and clean up any artefacts
        # This mode will not give a true lossless result because the resharpening and final temporal smooth
        # are still to come, but it will add further detail
        # However, it can introduce minor combing. This setting is best used together with source-match
        # it's effectively the final source-match stage)
        if self.lossless >= 2:
            lossed1 = self._make_lossless(match, inner_clip)
        else:
            lossed1 = match

        # --- Resharpen / retouch output

        # Resharpen to counteract temporal blurs.
        # Little sharpening needed for source-match mode since it has already recovered sharpness from source
        if self.sharp.sharp_mode <= 0:
            resharp = lossed1
        elif self.sharp.sharp_mode == 1:
            resharp = core.std.Expr(
                [lossed1, lossed1.std.Convolution(matrix=self._matrix)],
                expr=f"x x y - {self.sharp.sharp_adj} * +",
            )
        else:
            vresharp1 = core.std.Merge(
                lossed1.std.Maximum(coordinates=[0, 1, 0, 0, 0, 0, 1, 0]),
                lossed1.std.Minimum(coordinates=[0, 1, 0, 0, 0, 0, 1, 0]),
            )
            if self.precise:  # Precise mode: reduce tiny overshoot
                vresharp = core.std.Expr(
                    [vresharp1, lossed1],
                    expr="x y < x {i1} + x y > x {i1} - x ? ?".format(
                        i1=scale_value(1, 8, self._bit_depth)
                    ),
                )
            else:
                vresharp = vresharp1
            resharp = core.std.Expr(
                [lossed1, vresharp.std.Convolution(matrix=self._matrix)],
                expr=f"x x y - {self.sharp.sharp_adj} * +",
            )

        # Slightly thin down 1-pixel high horizontal edges that have been widened
        # into neighboring field lines by the interpolator
        sv_thin_sc = self.sharp.sharp_thin * 6.0
        if self.sharp.sharp_thin > 0:
            expr = f"y x - {sv_thin_sc} * {self._neutral} +"
            vert_med_d = core.std.Expr(
                [
                    lossed1,
                    lossed1.rgvs.VerticalCleaner(mode=1 if self._is_gray else [1, 0]),
                ],
                expr=expr if self._is_gray else [expr, ""],
            )
            vert_med_d = vert_med_d.std.Convolution(
                matrix=[1, 2, 1], planes=0, mode="h"
            )
            expr = (
                f"y {self._neutral} - abs x {self._neutral} - abs > y {self._neutral} ?"
            )
            neighbor_d = core.std.Expr(
                [vert_med_d, vert_med_d.std.Convolution(matrix=self._matrix, planes=0)],
                expr=expr if self._is_gray else [expr, ""],
            )
            thin = core.std.MergeDiff(resharp, neighbor_d, planes=0)
        else:
            thin = resharp

        # Back blend the blurred difference between sharpened & unsharpened clip,
        # before (1st) sharpness limiting (Sbb == 1,3).
        # A small fidelity improvement
        if self.sharp.sharp_back_blend not in [1, 3]:
            back_blend1 = thin
        else:
            back_blend1 = core.std.MakeDiff(
                thin,
                gauss_blur(
                    core.std.MakeDiff(thin, lossed1, planes=0).std.Convolution(
                        matrix=self._matrix, planes=0
                    ),
                    1.2,
                ),
                planes=0,
            )

        # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
        # Occurs here (before final temporal smooth) if SLMode == 1,2.
        # This location will restrict sharpness more, but any artefacts introduced will be smoothed
        if self.sharp.sharp_limit_mode == 1:
            if self.sharp.sharp_limit_rad <= 1:
                sharp_limit1 = core.rgvs.Repair(back_blend1, edi, mode=1)
            else:
                sharp_limit1 = core.rgvs.Repair(
                    back_blend1, core.rgvs.Repair(back_blend1, edi, mode=12), mode=1
                )
        elif self.sharp.sharp_limit_mode == 2:
            sharp_limit1 = mt_clamp(
                back_blend1,
                t_max,
                t_min,
                self._sharp_overshoot,
                self._sharp_overshoot,
            )
        else:
            sharp_limit1 = back_blend1

        # Back blend the blurred difference between sharpened & unsharpened clip, after (1st) sharpness limiting (Sbb == 2,3). A small fidelity improvement
        if self.sharp.sharp_back_blend < 2:
            back_blend2 = sharp_limit1
        else:
            back_blend2 = core.std.MakeDiff(
                sharp_limit1,
                gauss_blur(
                    core.std.MakeDiff(sharp_limit1, lossed1, planes=0).std.Convolution(
                        matrix=self._matrix, planes=0
                    ),
                    1.2,
                ),
                planes=0,
            )

        # Add back any extracted noise, prior to final temporal smooth -
        # this will restore detail that was removed as "noise" without restoring the noise itself.
        add_noise1 = self._grain_restore(
            back_blend2, final_noise, self.noise.grain_restore
        )

        # Final light linear temporal smooth for denoising
        if self.tr2 > 0:
            stable_super = add_noise1.mv.Super(
                sharp=self.me.sub_pel_interp, levels=1, **super_args
            )
        if self.tr2 <= 0:
            stable = add_noise1
        elif self.tr2 == 1:
            stable = core.mv.Degrain1(
                add_noise1,
                stable_super,
                self._b_vec1,
                self._f_vec1,
                thsad=self.me.th_sad2,
                thscd1=self.me.th_scd1,
                thscd2=self.me.th_scd2,
            )
        elif self.tr2 == 2:
            stable = core.mv.Degrain2(
                add_noise1,
                stable_super,
                self._b_vec1,
                self._f_vec1,
                self._b_vec2,
                self._f_vec2,
                thsad=self.me.th_sad2,
                thscd1=self.me.th_scd1,
                thscd2=self.me.th_scd2,
            )
        else:
            stable = core.mv.Degrain3(
                add_noise1,
                stable_super,
                self._b_vec1,
                self._f_vec1,
                self._b_vec2,
                self._f_vec2,
                self._b_vec3,
                self._f_vec3,
                thsad=self.me.th_sad2,
                thscd1=self.me.th_scd1,
                thscd2=self.me.th_scd2,
            )

        # Remove areas of difference between final output & basic interpolated image
        # that are not bob-shimmer fixes: repairs motion blur caused by temporal smooth
        if self.repair.repair2 <= 0:
            repair2 = stable
        else:
            repair2 = self._keep_only_bob_shimmer_fixes(
                stable, edi, self.repair.repair2, self.repair.rep_chroma
            )

        # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
        # Occurs here (after final temporal smooth) if SLMode == 3,4.
        # Allows more sharpening here, but more prone to introducing minor artefacts
        if self.sharp.sharp_limit_mode == 3:
            if self.sharp.sharp_limit_rad <= 1:
                sharp_limit2 = core.rgvs.Repair(repair2, edi, mode=1)
            else:
                sharp_limit2 = core.rgvs.Repair(
                    repair2, core.rgvs.Repair(repair2, edi, mode=12), mode=1
                )
        elif self.sharp.sharp_limit_mode >= 4:
            sharp_limit2 = mt_clamp(
                repair2,
                t_max,
                t_min,
                self._sharp_overshoot,
                self._sharp_overshoot,
            )
        else:
            sharp_limit2 = repair2

        # Lossless=1 - inject source fields into result and clean up inevitable artefacts.
        # Provided NoiseRestore=0.0 or 1.0, this mode will make the script result properly lossless,
        # but this will retain source artefacts and cause some combing
        # (where the smoothed deinterlace doesn't quite match the source)
        if self.lossless == 1:
            lossed2 = self._make_lossless(sharp_limit2, inner_clip)
        else:
            lossed2 = sharp_limit2

        # Add back any extracted noise, after final temporal smooth.
        # This will appear as noise/grain in the output.
        add_noise2 = self._grain_restore(lossed2, final_noise, self.noise.noise_restore)

        # --- Post-Processing

        # Shutter motion blur - get level of blur depending on output framerate and blur already in source
        blur_level = (
            (
                self.motion_blur.shutter_angle_out * self.fps_divisor
                - self.motion_blur.shutter_angle_src
            )
            * 100
            / 360
        )
        if blur_level < 0:
            raise vs.Error(
                "QTGMC: cannot reduce motion blur already in source: increase shutter_angle_out or fps_divisor"
            )
        if blur_level > 200:
            raise vs.Error(
                "QTGMC: exceeded maximum motion blur level: decrease shutter_angle_out or fps_divisor"
            )

        # ShutterBlur mode 2,3 - get finer resolution motion vectors to reduce blur "bleeding" into static areas
        r_block_divide = [1, 1, 2, 4][self.motion_blur.shutter_blur]
        r_block_size = max(self.me.block_size // r_block_divide, 4)
        r_overlap = max(self.me.overlap // r_block_divide, 2)
        r_block_divide = self.me.block_size // r_block_size
        r_lambda = self.me.coherence // (r_block_divide * r_block_divide)
        if self.motion_blur.shutter_blur > 1:
            recalculate_args = dict(
                thsad=self.me.th_sad1,
                blksize=r_block_size,
                overlap=r_overlap,
                search=self.me.search,
                searchparam=self.me.search_param,
                truemotion=self.me.true_motion,
                lambda_=r_lambda,
                pnew=self.me.penalty_new,
                dct=self.me.dct,
                chroma=self.me.chroma_motion,
            )
            sb_b_vec1 = core.mv.Recalculate(
                self._search_super, self._b_vec1, **recalculate_args
            )
            sb_f_vec1 = core.mv.Recalculate(
                self._search_super, self._f_vec1, **recalculate_args
            )
        elif self.motion_blur.shutter_blur > 0:
            sb_b_vec1 = self._b_vec1
            sb_f_vec1 = self._f_vec1

        # Shutter motion blur - use MFlowBlur to blur along motion vectors
        if self.motion_blur.shutter_blur > 0:
            sblur_super = add_noise2.mv.Super(
                sharp=self.me.sub_pel_interp, levels=1, **super_args
            )
            sblur = core.mv.FlowBlur(
                add_noise2,
                sblur_super,
                sb_b_vec1,
                sb_f_vec1,
                blur=blur_level,
                thscd1=self.me.th_scd1,
                thscd2=self.me.th_scd2,
            )

        # Shutter motion blur - use motion mask to reduce blurring in areas of low motion -
        # also helps reduce blur "bleeding" into static areas, then select blur type
        if self.motion_blur.shutter_blur <= 0:
            sblurred = add_noise2
        elif self.motion_blur.shutter_blur_limit <= 0:
            sblurred = sblur
        else:
            sb_motion_mask = core.mv.Mask(
                depth(self._search_clip, 8, dither_type=DitherType.NONE),
                self._b_vec1,
                kind=0,
                ml=self.motion_blur.shutter_blur_limit,
            )
            sb_motion_mask = depth(
                sb_motion_mask,
                self._search_clip.format.bits_per_sample,
                dither_type=DitherType.NONE,
            )
            sblurred = core.std.MaskedMerge(add_noise2, sblur, sb_motion_mask)

        # Reduce frame rate
        if self.fps_divisor > 1:
            decimated = sblurred.std.SelectEvery(cycle=self.fps_divisor, offsets=0)
        else:
            decimated = sblurred

        # Crop off temporary vertical padding
        if self.border:
            cropped = decimated.std.Crop(top=4, bottom=4)
        else:
            cropped = decimated

        # Show output of choice
        if self.noise.show_noise <= 0:
            output = cropped
        else:
            expr = f"x {self._neutral} - {self.noise.show_noise} * {self._neutral} +"
            output = final_noise.std.Expr(
                expr=(
                    expr
                    if self.noise.chroma_noise or self._is_gray
                    else [expr, repr(self._neutral)]
                )
            )
        output = output.std.SetFieldBased(value=0)

        return output

    def get_globals(self) -> QTGMCGlobals:
        return QTGMCGlobals(
            search_clip=self._search_clip,
            search_super=self._search_super,
            b_vec1=self._b_vec1,
            f_vec1=self._f_vec1,
            b_vec2=self._b_vec2,
            f_vec2=self._f_vec2,
            b_vec3=self._b_vec3,
            f_vec3=self._f_vec3,
        )

    def _build_search_clip(
        self,
        bobbed: vs.VideoNode,
        width: int,
        height: int,
    ) -> vs.VideoNode:
        # The bobbed clip will shimmer due to being derived from alternating fields.
        # Temporally smooth over the neighboring frames using a binomial kernel.
        # Binomial kernels give equal weight to even and odd frames and hence average away the shimmer.
        # The two kernels used are [1 2 1] and [1 4 6 4 1] for radius 1 and 2.
        # These kernels are approximately Gaussian kernels, which work well as a prefilter
        # before motion analysis (hence the original name for this script)
        #
        # Create linear weightings of neighbors first
        bobbed = scdetect(bobbed, 28 / 255)
        # -2    -1    0     1     2
        if self.tr0 > 0:
            # 0.00  0.33  0.33  0.33  0.00
            ts1 = bobbed.misc.AverageFrames(
                [1] * 3, scenechange=True, planes=self._cm_planes
            )
        if self.tr0 > 1:
            # 0.20  0.20  0.20  0.20  0.20
            ts2 = bobbed.misc.AverageFrames(
                [1] * 5, scenechange=True, planes=self._cm_planes
            )

        # Combine linear weightings to give binomial weightings - TR0=0: (1), TR0=1: (1:2:1), TR0=2: (1:4:6:4:1)
        if self.tr0 <= 0:
            binomial0 = bobbed
        elif self.tr0 == 1:
            binomial0 = core.std.Merge(
                ts1,
                bobbed,
                weight=0.25 if self.me.chroma_motion or self._is_gray else [0.25, 0],
            )
        else:
            binomial0 = core.std.Merge(
                core.std.Merge(
                    ts1,
                    ts2,
                    weight=(
                        0.357 if self.me.chroma_motion or self._is_gray else [0.357, 0]
                    ),
                ),
                bobbed,
                weight=0.125 if self.me.chroma_motion or self._is_gray else [0.125, 0],
            )

        # Remove areas of difference between temporal blurred motion search clip and bob
        # that are not due to bob-shimmer - removes general motion blur
        if self.repair.repair0 > 0:
            repair0 = self._keep_only_bob_shimmer_fixes(
                binomial0, bobbed, self.repair.rep_chroma and self.me.chroma_motion
            )
        else:
            repair0 = binomial0

        if self.me.search_clip_pp == 1:
            spatial_blur = (
                repair0.resize.Bilinear(width // 2, height // 2)
                .std.Convolution(matrix=self._matrix, planes=self._cm_planes)
                .resize.Bilinear(width, height)
            )
        elif self.me.search_clip_pp >= 2:
            spatial_blur = gauss_blur(
                repair0.std.Convolution(matrix=self._matrix, planes=self._cm_planes),
                1.75,
            )
            spatial_blur = core.std.Merge(
                spatial_blur,
                repair0,
                weight=0.1 if self.me.chroma_motion or self._is_gray else [0.1, 0],
            )

        if self.me.search_clip_pp <= 0:
            search_clip = repair0
        elif self.me.search_clip_pp < 3:
            search_clip = spatial_blur
        else:
            expr = "x {i3} + y < x {i3} + x {i3} - y > x {i3} - y ? ?".format(
                i3=scale_value(3, 8, self._bit_depth)
            )
            tweaked = core.std.Expr(
                [repair0, bobbed],
                expr=expr if self.me.chroma_motion or self._is_gray else [expr, ""],
            )
            expr = "x {i7} + y < x {i2} + x {i7} - y > x {i2} - x 51 * y 49 * + 100 / ? ?".format(
                i7=scale_value(7, 8, self._bit_depth),
                i2=scale_value(2, 8, self._bit_depth),
            )
            search_clip = core.std.Expr(
                [spatial_blur, tweaked],
                expr=expr if self.me.chroma_motion or self._is_gray else [expr, ""],
            )
        search_clip = prefilter_to_full_range(
            search_clip, self.strength, self._cm_planes
        )
        if self._bit_depth > 8 and self.me.fast_ma:
            search_clip = depth(search_clip, 8, dither_type=DitherType.NONE)

        return search_clip

    def _keep_only_bob_shimmer_fixes(
        self,
        clip: vs.VideoNode,
        ref: vs.VideoNode,
        repair: int = 1,
        chroma: bool = True,
    ) -> vs.VideoNode:
        # TODO
        pass

    def _generate_2nd_field_noise(
        self, noise: vs.VideoNode, denoised: vs.VideoNode
    ) -> vs.VideoNode:
        # TODO
        pass

    def _interpolate(
        self,
        edi_input: vs.VideoNode,
        bobbed: vs.VideoNode,
        edi_mode: EdiMethod,
        nn_size: int,
        num_neurons: int,
        edi_qual: int,
        edi_max_dist: int,
    ) -> vs.VideoNode:
        # TODO
        pass

    def _apply_source_match(
        self,
        deint: vs.VideoNode,
        source: vs.VideoNode,
        hpad: int,
        vpad: int,
    ) -> vs.VideoNode:
        # TODO
        # self._b_vec1 if self.max_tr > 0 else None,
        # self._f_vec1 if self.max_tr > 0 else None,
        # self._b_vec2 if self.max_tr > 1 else None,
        # self._f_vec2 if self.max_tr > 1 else None,
        pass

    def _make_lossless(
        self, input: vs.VideoNode, source: vs.VideoNode
    ) -> vs.VideoNode:
        # TODO
        pass

    def _grain_restore(
        self, input: vs.VideoNode, final_noise: vs.VideoNode, strength: float
    ) -> vs.VideoNode:
        if self.noise.grain_restore <= 0:
            return input
        else:
            # Average luma of FFT3DFilter extracted noise is 128.5, so deal with that too
            expr = f"x {self._noise_center} - {strength} * {self._neutral} +"
            return core.std.MergeDiff(
                input,
                final_noise.std.Expr(
                    expr=(
                        expr if self.noise.chroma_noise or self._is_gray else [expr, ""]
                    )
                ),
                planes=self._cn_planes,
            )


# FIXME: Is there a function for this in JET?
def scdetect(clip: vs.VideoNode, threshold: float = 0.1) -> vs.VideoNode:
    def _copy_property(_n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props["_SceneChangePrev"] = f[1].props["_SceneChangePrev"]
        fout.props["_SceneChangeNext"] = f[1].props["_SceneChangeNext"]
        return fout

    assert check_variable(clip, scdetect)

    sc = clip
    if clip.format.color_family == vs.RGB:
        sc = clip.resize.Point(format=vs.GRAY8, matrix_s="709")

    sc = sc.misc.SCDetect(threshold)
    if clip.format.color_family == vs.RGB:
        sc = clip.std.ModifyFrame([clip, sc], _copy_property)

    return sc


# FIXME: Is there a function for this in JET?
def mt_clamp(
    clip: vs.VideoNode,
    bright: vs.VideoNode,
    dark: vs.VideoNode,
    overshoot: int = 0,
    undershoot: int = 0,
    planes: PlanesT = None,
) -> vs.VideoNode:
    """clamp the value of the clip between bright + overshoot and dark - undershoot"""
    check_ref_clip(clip, bright, mt_clamp)
    check_ref_clip(clip, dark, mt_clamp)
    planes = normalize_planes(clip, planes)

    if complexpr_available:
        expr = f"x z {undershoot} - y {overshoot} + clamp"
    else:
        expr = f"x z {undershoot} - max y {overshoot} + min"
    return norm_expr([clip, bright, dark], expr, planes)
