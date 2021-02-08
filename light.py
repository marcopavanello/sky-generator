# Libraries
import functions as fun
import numpy as np
from constants import EARTH_RADIUS, SUN_IRRADIANCE, MIE_COEFFICIENT, NUM_WAVELENGTHS, OZONE_COEFFICIENT, RAYLEIGH_COEFFICIENT
from math import cos, exp, pi, radians, sin, sqrt, dist
from properties import air_density, altitude, dust_density, ozone_density, steps, sun_lat
from random import uniform as random


# Definitions
# convert altitude from km to m and clamp to avoid intersection issues
cam_altitude = 1000 * max(min(altitude, 59.999), 0.001)
camera_position = np.array([0, 0, EARTH_RADIUS + cam_altitude])
# convert sun latitude and longitude to vector
sun_direction = fun.geographical_to_direction(radians(sun_lat), 0)
# scattering and absorption coefficients
coefficients = np.array(
    [RAYLEIGH_COEFFICIENT, 1.11 * MIE_COEFFICIENT, OZONE_COEFFICIENT], dtype=object)
density_multipliers = np.array([air_density, dust_density, ozone_density])


def ray_optical_depth(ray_origin, ray_direction, ray_length):
    # step along the ray in segments and accumulate the optical depth along each segment
    segment_length = ray_length / steps
    segment = segment_length * ray_direction
    optical_depth = np.zeros(3)
    # the density of each segment is evaluated at its middle
    middle_point = ray_origin + 0.5 * segment

    for _ in range(steps):
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
    '''
    still using 1 single sample because it still gives errors,
    when ready we can use multiple samples per pixel as Monte Carlo method.
    the algorithm is (correct me if i'm wrong):
    - start ray from camera, end at random position, accumulate light from sun -> end point -> camera
    - then shoot a ray at random direction from previous end point, and accumulate sun -> new end point -> old end point -> camera
    - continue till the bounces end
    '''
    ray_origin = camera_position
    throughput = 0

    # light bounces per sample
    for _ in range(4):
        # if Earth surface isn't hit then hit atmosphere (one or the other needs to be hit)
        distance = fun.surface_intersection(ray_origin, ray_direction)
        if distance < 0:
            distance = fun.atmosphere_intersection(ray_origin, ray_direction)
        # pick a random point between origin and end of ray
        ray_length = random(0, 1) * distance
        # send ray to sun, it has to end in atmosphere otherwise return black (?)
        ray_end = ray_origin + ray_direction * ray_length
        ray_length_sun = fun.surface_intersection(ray_end, sun_direction)
        if ray_length_sun < 0:
            ray_length_sun = fun.atmosphere_intersection(
                ray_end, sun_direction)
        else:
            break
        # calculate optical depths
        optical_depth = ray_optical_depth(
            ray_origin, ray_direction, ray_length)
        optical_depth_sun = ray_optical_depth(
            ray_end, sun_direction, ray_length_sun)
        # attenuation of light
        transmittance = np.exp(-np.sum(coefficients * optical_depth))
        transmittance_sun = np.exp(-np.sum(coefficients * optical_depth_sun))
        # phase function (sr^-1)
        mu = np.dot(ray_direction, sun_direction)
        phase_R = fun.phase_rayleigh(mu)
        phase_M = fun.phase_mie(mu)
        # densities on scattering point
        height = sqrt(np.dot(ray_end, ray_end)) - EARTH_RADIUS
        density_R = fun.density_rayleigh(height)
        density_M = fun.density_mie(height)
        # compute scattering
        scattering_R = coefficients[0] * density_R * phase_R
        scattering_M = coefficients[1] * density_M * phase_M
        scattering = scattering_R * phase_R + scattering_M * phase_M

        # change ray direction for a new path
        ray_direction = np.array(
            [random(-1, 1), random(-1, 1), random(-1, 1)])

        # accumulate light
        throughput += SUN_IRRADIANCE * transmittance * scattering * transmittance_sun

    return throughput
