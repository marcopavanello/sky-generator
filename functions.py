# Libraries
import numpy as np
from constants import ATMOSPHERE_RADIUS, COLOR_MATCHING_FUNCTIONS, EARTH_RADIUS, FILMIC_LOOK, ILLUMINANT_D65, MIE_G, MIE_SCALE, RAYLEIGH_SCALE, SQR_G, WAVELENGTHS_STEP
from math import cos, exp, pi, sin, sqrt


# Functions
def density_rayleigh(height):
    return exp(-height / RAYLEIGH_SCALE)


def density_mie(height):
    return exp(-height / MIE_SCALE)


def density_ozone(height):
    if height < 10e3 or height >= 40e3:
        return 0
    elif height >= 10e3 and height < 25e3:
        return 1 / 15e3 * height - 2 / 3
    else:
        return -(1 / 15e3 * height - 8 / 3)


def phase_rayleigh(mu):
    return 3 / (16 * pi) * (1 + mu * mu)


def phase_mie(mu):
    return (3 * (1 - SQR_G) * (1 + mu * mu)) / (8 * pi * (2 + SQR_G) * ((1 + SQR_G - 2 * MIE_G * mu)**1.5))


def geographical_to_direction(lat, lon):
    return np.array([cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)])


def atmosphere_intersection(ray_origin, ray_direction):
    b = -2 * np.dot(ray_direction, -ray_origin)
    c = np.dot(ray_origin, ray_origin) - ATMOSPHERE_RADIUS * ATMOSPHERE_RADIUS
    t = (-b + sqrt(b * b - 4 * c)) / 2
    return ray_origin + ray_direction * t


def surface_intersection(ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin)
    c = np.dot(ray_origin, ray_origin) - EARTH_RADIUS * EARTH_RADIUS
    discriminant = b * b - 4 * c
    if discriminant < 0:
        return -1
    else:
        t = (-b - sqrt(discriminant)) / 2
        if t > 0:
            return t
        t = (-b + sqrt(discriminant)) / 2
        if t > 0:
            return t
        else:
            return -1


def spectrum_to_xyz(spectrum):
    # integral
    sum = np.sum(spectrum[:, np.newaxis] * COLOR_MATCHING_FUNCTIONS, axis=0)
    return sum * WAVELENGTHS_STEP


def xyz_to_rgb(xyz, exposure):
    # XYZ to sRGB linear
    sRGB_linear = np.dot(ILLUMINANT_D65, xyz)
    # apply exposure
    sRGB_exp = sRGB_linear * 2**exposure
    # avoid negative values
    sRGB_1 = np.maximum(1e-5, sRGB_exp)
    # apply filmic log encoding
    sRGB_log = (np.log2(sRGB_1 / 0.18) + 10) / 16.5
    # clamp sRGB between 0 and 1
    sRGB_2 = np.clip(sRGB_log, 1e-5, 1)
    # apply look contrast
    index = np.array(sRGB_2 * 4095, np.int)
    return np.array([FILMIC_LOOK[i] for i in index])
