import vangenuchten as vg

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.integrate import cumulative_trapezoid as cumtrapz


def smoothstep(x):
    """
    A cubic 'smooth step' polynomial.
    """
    x = np.asarray(x)
    result = np.empty_like(x)

    zero = x <= 0
    one = x >= 1
    interm = np.logical_not(np.logical_or(zero, one))

    result[zero] = 0.0
    result[one] = 1.0
    result[interm] = 3 * x[interm] ** 2 - 2 * x[interm] ** 3

    return result


def smoothceiling(x):
    """
    Like 'smooth step', but does not saturate at the low end.
    """
    return np.where(np.asarray(x) >= 0.5, smoothstep(x), 1.5 * x - 0.25)


def smoothfloor(x):
    """
    Like 'smooth step', but does not saturate at the high end.
    """
    return np.where(np.asarray(x) <= 0.5, smoothstep(x), 1.5 * x - 0.25)


def smoothclip(x, center, width, ceiling=True, floor=True, **kwargs):
    """
    An S-shaped clipping function with a given center and width, based on `smoothceiling`. The lower and upper clips
    can be disabled by setting `ceiling` or `floor`, respectively, to false.
    """
    if ceiling and floor:
        f = smoothstep
    elif ceiling:
        f = smoothceiling
    elif floor:
        f = smoothfloor

    return width * (f(2.0 * (x - center) / (3.0 * width) + 0.5) - 0.5) + center


def centers_widths(threshold_h):
    """
    Calculate the centers and widths, in $\log(h)$-space, of pressure-head ranges separated by `threshold_h`.
    """
    centers = np.exp(uniform_filter1d(np.log(threshold_h), 2, origin=-1)[:-1])  # 2 is the length of the window, giving
    # a pairwise average.

    widths = np.diff(np.log(threshold_h))

    # Centers of the first and last class are calculated by taking their widths to be the same as those of neighboring
    # classes
    center_smallest = np.exp((np.log(centers[0]) - widths[0]))

    center_largest = np.exp((np.log(centers[-1]) + widths[-1]))

    return centers, widths, center_smallest, center_largest


def clipping(h, threshold_h):
    """
    Soft clips an array `h` to the the pressure-head ranges delineated by `threshold_h`. If `h` is $N$-dimensional, the
    result is $(N+1)$-dimensional with the first dimension corresponding to the different classes.
    """
    centers, widths, center_smallest, center_largest = centers_widths(threshold_h)

    h = np.atleast_1d(h)

    assert centers.ndim == widths.ndim == 1  # Using np.concatenate assumes there's only one dimension for the classes

    h_axes = (np.newaxis,) + (...,)  # new class dimension + existing h dimensions
    class_axes = (...,) + (np.newaxis,) * h.ndim  # existing class dimension + as many new axes as h has dimensions

    return np.concatenate(
        (
            # What follows is a single formula, but applied three times: to the smalest class, the intermediate classes,
            # and the largest class
            np.exp(smoothclip(np.log(h), np.log(center_smallest), widths[0], floor=False))[h_axes],  # We use widths[0]
            # here, because we take the width of the smallest class to be the same as that of the second class
            np.exp(smoothclip(np.log(h[h_axes]), np.log(centers[class_axes]), widths[class_axes])),
            np.exp(smoothclip(np.log(h), np.log(center_largest), widths[-1], ceiling=False))[
                h_axes
            ],  # We use widths[-1]
            # here, because we take the width of the largest class to be the same as that of the penultimate class
        )
    )


def comps(h, threshold_h, a, n, mcr, mcs, **kwargs):
    """
    Divides a Van Genuchten retention curve into smooth segments separated by `threshold_h` values of pressure head.

    Soil parameters (a, n, mcr, mcs) can be multidimensional, but must be manually broadcasted to have the same shape.
    The result has these 'soil' dimensions first (at least one, even if the soil parameters are 0-dimensional), followed
    by a dimension for the pressure-head class, and then the original dimensions of `h`.
    """
    h = np.atleast_1d(h)
    a = np.atleast_1d(a)
    n = np.atleast_1d(n)
    mcr = np.atleast_1d(mcr)
    mcs = np.atleast_1d(mcs)

    assert a.ndim == n.ndim == mcr.ndim == mcs.ndim

    clip_axes = (np.newaxis,) * a.ndim + (...,)  # new soil dimensions + existing h and class dimensions
    soil_axes = (...,) + (np.newaxis,) * (h.ndim + 1)  # existing soil dimensions + new h and class dimensions

    return vg.mc(clipping(h, threshold_h)[clip_axes], a[soil_axes], n[soil_axes], mcr[soil_axes], mcs[soil_axes])


def mc(sVG_comp, threshold_h, a, n, mcr, mcs, **kwargs):
    """
    Transforms Van Genuchten retention curve segments into proper retention sub-curves by calculating and subtracting
    their residual water contents.

    The `sVG_comp` passed to this function is normaly obtained using the `comps` function. We assume here that the first
    two dimensions of `sVG_comp` correspond to soil type and size class, respectively. More than one dimension for soil
    type will not work.
    """
    a = np.atleast_1d(a)
    n = np.atleast_1d(n)
    mcr = np.atleast_1d(mcr)
    mcs = np.atleast_1d(mcs)

    h_dims = sVG_comp.ndim - 2

    return (
        sVG_comp
        - np.hstack(
            (
                mcr[:, None],  # We subtract mcr from the first segment
                # We put the mcr values along the first dimension, which corresponds to soil type
                vg.mc(
                    threshold_h[None, : sVG_comp.shape[1] - 1], a[:, None], n[:, None], mcr[:, None], mcs[:, None]
                ),  # From the following segments, we subtract the moisture content at the dry end of the class.
                # :sVG_comp.shape[1]-1 is for compatibility with legacy code in which threshold_h could be as long as
                # the number of segments in sVG_comp. Normally it is one element shorter and this slice doesn't do
                # anything.
            )
        )[(...,) + h_dims * (np.newaxis,)]
    )


def smoothstep_prime(x):
    """
    Derivative of the cubic 'smooth step' polynomial.
    """
    x = np.asarray(x)

    return np.where(np.logical_or(x <= 0, x >= 1), 0.0, 6 * x * (1 - x))


def smoothceiling_prime(x):
    """
    Derivative of the cubic 'smooth ceiling' function.
    """
    return np.where(np.asarray(x) >= 0.5, smoothstep_prime(x), 1.5)


def smoothfloor_prime(x):
    """
    Derivative of the cubic 'smooth floor' function.
    """
    return np.where(np.asarray(x) <= 0.5, smoothstep_prime(x), 1.5)


def smoothclip_prime(x, center, width, ceiling=True, bottom=True, **kwargs):
    """
    Derivative of the `smoothclip` clipping function.
    """
    if ceiling and bottom:
        f = smoothstep_prime
    elif ceiling:
        f = smoothceiling_prime
    elif bottom:
        f = smoothfloor_prime

    return f(2.0 * (x - center) / (3.0 * width) + 0.5) * 2 / 3


def clipping_prime(h, clipping_h, threshold_h):
    """
    Calculate the derivative of the soft clipping performed by the`clipping` function.

    Since the `clipping` function is exponential, its derivative is the function itself times a derivative of the
    exponent. We take the former in as an argument to not recalculate it, and multiply it by the latter here to compute
    the full derivative.

    This function follows largely the same pattern as `clipping`, so all the comments made there apply here as well.
    """
    centers, widths, center_smallest, center_largest = centers_widths(threshold_h)

    h = np.atleast_1d(h)

    h_axes = (np.newaxis,) + (...,)
    class_axes = (...,) + (np.newaxis,) * h.ndim

    return np.concatenate(
        (
            clipping_h[0, ...]
            * smoothclip_prime(np.log(h), np.log(center_smallest), widths[0], bottom=False)[None, ...]
            / h[h_axes],
            clipping_h[1:-1, ...]
            * smoothclip_prime(np.log(h[h_axes]), np.log(centers[class_axes]), widths[class_axes])
            / h[h_axes],
            clipping_h[-1, ...]
            * smoothclip_prime(np.log(h), np.log(center_largest), widths[-1], ceiling=False)[h_axes]
            / h[h_axes],
        )
    )


def dmcdh_int(h, threshold_h, **kwargs):
    """
    Integrate the pressure-head derivatives of the Van Genuchten retention sub-curves over log pressure head.

    We assume here that `h` are positive and increasing.

    The dimensions of the result are: [class, (dimensions of `h`)]
    """
    h_clip = clipping(h, threshold_h)

    dmcdh = vg.dmcdh(h_clip, **kwargs) * clipping_prime(h, h_clip, threshold_h)

    # To do the integral, we first switch h to negative-increasing (the two internal flips), integrate, and switch the
    # result back to positive-increasing h
    return np.flip(cumtrapz(np.flip(dmcdh), -np.flip(np.log(h))))
