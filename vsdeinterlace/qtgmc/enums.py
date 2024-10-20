__all__ = ["EdiMethod", "DenoiseMethod", "DeintMethod", "InputType"]


class EdiMethod:
    """EEDI method to use within QTGMC"""

    BOB = 0
    BWDIF = 1
    EEDI3 = 2
    EEDI3_PLUS_NNEDI3 = 3
    NNEDI3 = 4


class DenoiseMethod:
    """Denoiser to use within QTGMC"""

    NONE = 0
    FFT3DF = 1
    DFTTEST = 2
    BM3D = 3
    KNLMeans = 4


class DeintMethod:
    """Deinterlacer to use within QTGMC"""

    NONE = 0
    BOB = 1
    DOUBLE_WEAVE = 2
    GENERATE = 3


class InputType:
    """Type of input being processed by QTGMC"""

    INTERLACED = 0
    """For interlaced input"""
    PROGRESSIVE = 1
    """For normal progressive input, useful for deshimmering"""
    PROGRESSIVE_WITH_COMBING = 2
    """For badly deinterlaced progressive input that has visible combing"""
