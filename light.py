# Libraries
import functions as fun
import numpy as np
from constants import EARTH_RADIUS, IRRADIANCE, MIE_COEFFICIENT, NUM_WAVELENGTHS, OZONE_COEFFICIENT, RAYLEIGH_COEFFICIENT
from math import cos, exp, pi, radians, sin, sqrt, dist
from properties import air_density, altitude, dust_density, ozone_density, steps, steps_light, sun_lat
from random import uniform as random


# Definitions
# convert altitude from km to m and clamp to avoid intersection issues
cam_altitude = 1000 * max(min(altitude, 59.999), 0.001)
cam_pos = np.array([0, 0, EARTH_RADIUS + cam_altitude])
# convert sun latitude and longitude to vector
sun_dir = fun.geographical_to_direction(radians(sun_lat), 0)
coefficients = np.array([RAYLEIGH_COEFFICIENT, 1.11 *
                         MIE_COEFFICIENT, OZONE_COEFFICIENT], dtype=np.object)
density_multipliers = np.array([air_density, dust_density, ozone_density])


def single_scattering(ray_dir):
    # intersection between camera and top of atmosphere
    end_point = fun.atmosphere_intersection(cam_pos, ray_dir)
    # distance from camera to top of atmosphere
    ray_length = dist(cam_pos, end_point)
    # to compute the inscattering, we step along the ray in segments and
    # accumulate the inscattering as well as the optical depth along each segment
    segment_length = ray_length / steps
    segment = segment_length * ray_dir
    optical_depth = np.zeros(3)
    spectrum = np.zeros(NUM_WAVELENGTHS)
    # cosine of angle between camera and sun
    mu = np.dot(ray_dir, sun_dir)
    # phase functions (sr^-1)
    phase_function_R = fun.phase_rayleigh(mu)
    phase_function_M = fun.phase_mie(mu)
    # the density and in-scattering of each segment is evaluated at its middle
    middle_point = cam_pos + 0.5 * segment

    for _ in range(steps):
        # height above sea level
        height = sqrt(np.dot(middle_point, middle_point)) - EARTH_RADIUS
        # evaluate and accumulate optical depth along the ray
        densities = np.array([fun.density_rayleigh(
            height), fun.density_mie(height), fun.density_ozone(height)])
        density = density_multipliers * densities
        optical_depth += density * segment_length

        # if the Earth isn't in the way, evaluate inscattering from the sun
        if not fun.surface_intersection(middle_point, sun_dir):
            optical_depth_light = ray_optical_depth(middle_point, sun_dir)
            # attenuation of light
            extinction_density = (
                optical_depth + density_multipliers * optical_depth_light) * coefficients
            attenuation = np.exp(-np.sum(extinction_density))
            scattering_density_R = density[0] * RAYLEIGH_COEFFICIENT
            scattering_density_M = density[1] * MIE_COEFFICIENT
            # compute spectrum
            spectrum += attenuation * \
                (phase_function_R * scattering_density_R +
                 phase_function_M * scattering_density_M)

        # advance along ray
        middle_point += segment

    # spectrum at pixel in radiance (W*m^-2*nm^-1*sr^-1)
    return IRRADIANCE * spectrum * segment_length


def ray_optical_depth(ray_origin, ray_dir):
    # optical depth along a ray through the atmosphere
    end_point = fun.atmosphere_intersection(ray_origin, ray_dir)
    ray_length = dist(ray_origin, end_point)
    # step along the ray in segments and accumulate the optical depth along each segment
    segment_length = ray_length / steps_light
    segment = segment_length * ray_dir
    optical_depth = np.zeros(3)
    # the density of each segment is evaluated at its middle
    middle_point = ray_origin + 0.5 * segment

    for _ in range(steps_light):
        # height above sea level
        height = sqrt(np.dot(middle_point, middle_point)) - EARTH_RADIUS
        # accumulate optical depth of this segment
        density = np.array([fun.density_rayleigh(
            height), fun.density_mie(height), fun.density_ozone(height)])
        optical_depth += density
        # advance along ray
        middle_point += segment

    return optical_depth * segment_length


def multiple_scattering(ray_direction):
    ray_origin = cam_pos

    for bounce in range(5):
        distance = fun.surface_intersection(ray_origin, ray_direction)
        # if Earth surface isn't hit, calculate atmosphere
        if distance < 0:
            distance = fun.atmosphere_intersection(ray_origin, ray_direction)
        # pick a random point between origin and end of distance
        ray_end = random(0, 1) * distance
        optical_depth = ray_optical_depth(ray_origin, ray_direction)
        # advance to random scattering event
        ray_direction = np.array(
            [random(-1, 1), random(-1, 1), random(-1, 1)])

    return IRRADIANCE
