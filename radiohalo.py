import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time

# ===============CONSTANTS===============
tau = 2*np.math.pi
# Windows python can only support a smallest radius to 1e-11 before losing precision. Actual radius has exponent of -15
alphaRadius = 1.67824e-9/2 # Radius of alpha particle 1.67824(83) femtometers
biotiteFlakeThickness = 3e-6
visibility_dose = 1.5e13 # alpha particles/cm2
overexposure_dose = 1e14 # alpha particles/cm2 Currently a WAG. need data
bleach_dose = 1e15 # alpha particles/cm2 Currently a WAG. need data

radiumAlphaDecays = [
    ('238U', 4.26, colors.to_rgb('k'),                  '', 0., 0., colors.to_rgb('k')),
    ('234U', 4.85, colors.to_rgb('xkcd:electric green'),'', 0., 0., colors.to_rgb('k')),
    ('230Th', 4.76, colors.to_rgb('xkcd:spruce'),       '', 0., 0., colors.to_rgb('k')),
    ('226Ra', 4.87, colors.to_rgb('xkcd:cerise'),       '', 0., 0., colors.to_rgb('k')),
    ('222Rn', 5.59, colors.to_rgb('xkcd:burnt orange'), '', 0., 0., colors.to_rgb('k')),
    ('218Po', 6.11, colors.to_rgb('xkcd:lavender'),     '218At', 2e-4, 6.87, colors.to_rgb('k')),
    ('214Po', 7.83, colors.to_rgb('r'),                 '214Bi', 2e-4, 5.62, colors.to_rgb('k')),
    ('210Po', 5.40, colors.to_rgb('b'),                 '210Bi', 1.3e-6, 5.03, colors.to_rgb('k'))
]

actiniumAlphaDecays = [
    ('235U',  4.67, colors.to_rgb('k'),                  '', 0., 0., colors.to_rgb('k')),
    ('231Pa', 5.14, colors.to_rgb('xkcd:spruce'),        '', 0., 0., colors.to_rgb('k')),
    ('227Th', 6.14, colors.to_rgb('xkcd:cerise'),        '227Ac', 0.0138, 5.04, colors.to_rgb('k')),
    ('223Ra', 5.97, colors.to_rgb('xkcd:electric green'),'223Fr', 8.28e-5, 5.56, colors.to_rgb('k')),
    ('219Rn', 6.94, colors.to_rgb('xkcd:burnt orange'),  '219At', 7.75e-5, 6.34, colors.to_rgb('k')),
    ('215Po', 7.52, colors.to_rgb('r'),                  '215At', 2.3e-6, 8.17, colors.to_rgb('k')),
    ('211Bi', 6.75, colors.to_rgb('b'),                  '211Po', 2.8e-3, 7.59, colors.to_rgb('k'))
]

thoriumAlphaDecays = [
    ('232Th', 4.08, colors.to_rgb('k'),                  '', 0., 0., colors.to_rgb('k')),
    ('228Th', 5.52, colors.to_rgb('xkcd:spruce'),        '', 0., 0., colors.to_rgb('k')),
    ('224Ra', 5.78, colors.to_rgb('xkcd:cerise'),        '', 0., 0., colors.to_rgb('k')),
    ('220Rn', 6.4, colors.to_rgb('xkcd:burnt orange'),   '', 0., 0., colors.to_rgb('k')),
    ('216Po', 6.9, colors.to_rgb('b'),                   '', 0., 0., colors.to_rgb('k')),
    ('212Po', 8.95, colors.to_rgb('r'),                  '212Bi', 0.3594, 6.20, colors.to_rgb('xkcd:lavender'))
]

# Current conversion factor 7.83 MeV corresponds to 28 microns radius
radiusNormalizationFactor = 280000 # MeV/m Conversion between physical units and the assumed energy level distances. Needs to be calibrated

# ===============CONTROLS===============

decayList = radiumAlphaDecays[:]
iterations = 10000
colorify = True
flakeNumber = 0
alternateDecays = True
seed = None

# ===============SIMULATION===============

start_time = time.time()
print('Start: {}'.format(time.asctime(time.localtime(start_time))))

_, energies, colors, _, ratios, alt_energies, alt_colors = zip(*decayList)

maxHaloRadius = np.max(energies)/radiusNormalizationFactor # in micrometers

def solidAngle(sphere_radius, cap_radius):
    return tau * (1. - np.sqrt(1. - np.square(cap_radius/sphere_radius)))

def area2rho(area):
    return 2*np.arcsin(np.sqrt(area/(2*tau))) # Spherical radius of cap

def rad2rho(sphere_radius, cap_radius):
    return np.arcsin(cap_radius/sphere_radius)

def intersectionMagnitude(rho, separation):
    return np.maximum(0., 1. - separation/(2. * rho))

def greatCircleDistance(normal1, normal2):
    return np.arctan(np.norm(np.cross(normal1, normal2))/np.dot(normal1, normal2))

def capIntersection(rho, separation):
    # Solution from https://math.stackexchange.com/questions/45788/calculate-the-area-on-a-sphere-of-the-intersection-of-two-spherical-caps

    # spherical cap sector
    intersect_length = np.arccos(np.cos(rho)/np.cos(separation/2))
    alpha = 2 * np.arcsin(np.sin(intersect_length)/np.sin(rho))
    beta = np.arctan(1/(np.tan(alpha/2) * np.cos(rho)))

    return np.maximum(0., 2 * (2 * alpha * np.square(np.sin(rho/2)) - (alpha + 2 * beta - tau/2))) # Two spherical cap sectors less two sperical triangles

def intersectionLUT(lut_precision):
    rho = area2rho(1)
    lut_separation = np.linspace(0, 2*rho, 10**lut_precision+1)
    lut_ratios = np.around(intersectionMagnitude(rho, lut_separation), decimals=lut_precision)
    overlap_lut = capIntersection(area2rho(1), lut_separation)
    lut = dict(zip(list(lut_ratios), overlap_lut))

    def lookup(area, separation):
        return np.array([area * lut[np.round(intersectionMagnitude(area2rho(area), x), decimals=lut_precision)] for x in separation])
    
    return lookup

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / l2#np.expand_dims(l2, axis)

def uniformSphericalGenerator(n = 10, **kwargs):
    generator = np.random.default_rng(**kwargs)

    for _ in range(n):
        yield normalized(np.array([generator.normal(), generator.normal(), generator.normal()]))

# Generating seed according to the instructions so that we can have it for printing
run_seed = np.random.SeedSequence(seed)
print('seed = {}'.format(run_seed.entropy))

trials = iterations * len(decayList) # For each kind of radiohalo, all decays after the start must be completed for the total number of iterations
broadcast_shape = (iterations, len(decayList))
color_broadcast_shape = (iterations, len(decayList), 3)

raw_points = np.empty(trials * 3, dtype=float).reshape(trials, 3) # Preallocate array for random numbers
for i, x in enumerate(uniformSphericalGenerator(trials, seed=run_seed.spawn(1)[0])):
    raw_points[i,:] = x # Copy random points into preallocated array rows

energies_normalized = np.array(energies)/radiusNormalizationFactor # Normalize all halo distances so that the largest is at 1
# Create multiple copies of the energies until it there are 'iterations' number of copies. First broadcast to a compatible broadcasting shape, then reshape
main_energy_multiples = np.broadcast_to(energies_normalized, broadcast_shape)
main_color_broadcast = np.broadcast_to(np.array(colors, dtype=object), color_broadcast_shape)

# print(energy_multiples)

if alternateDecays:
    varigation_generator = np.random.default_rng(seed=run_seed.spawn(1)[0])
    varigation_trials = varigation_generator.uniform(size=broadcast_shape)
    broadcast_ratios = np.broadcast_to(ratios, broadcast_shape)
    alt_energies_normalized = np.array(alt_energies)/radiusNormalizationFactor
    alt_energy_multiples = np.broadcast_to(alt_energies_normalized, broadcast_shape)
    energy_multiples = np.where(varigation_trials > broadcast_ratios, main_energy_multiples, alt_energy_multiples).reshape(trials)

    alt_color_broadcast = np.broadcast_to(np.array(alt_colors, dtype=object), color_broadcast_shape)
    color_condition = np.tile((varigation_trials > broadcast_ratios).reshape(iterations, len(decayList), 1), (1,1,3))
    color_broadcast = np.where(np.broadcast_to(color_condition, color_broadcast_shape), main_color_broadcast, alt_color_broadcast).reshape(trials, 3)

else:
    energy_multiples = main_energy_multiples.reshape(trials)
    color_broadcast = main_color_broadcast.reshape(trials, 3)

points = (energy_multiples * raw_points.T).T # Double transpose required for vertical vector to row multiplication. Very little time cost
slice = points[np.abs(points[:,2]) - flakeNumber*biotiteFlakeThickness < biotiteFlakeThickness]

# Choosing colors for alternate decays
point_colors = color_broadcast if colorify else 'xkcd:chocolate brown'


# lookup = intersectionLUT(5)

# compare_size = 3
# compare_sep = np.linspace(0, 2*area2rho(compare_size), 100, endpoint=False)
# comparison = capIntersection(area2rho(compare_size), compare_sep)
# precomputed = lookup(compare_size, compare_sep)
# print((comparison - precomputed)/comparison)
# TODO, this lookup process seems to be flawed somehow

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(compare_sep+compare_sep, comparison+precomputed, c='b')

# ==================== PLOTS ==========================================

def spherePlot():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', proj_type='ortho')
    # ax.set_box_aspect((2*maxHaloRadius, 2*maxHaloRadius, 2*maxHaloRadius))

    # Do the same magic with colors as with decay energy to colorize each decay specially
    ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c=point_colors, alpha=0.5)
    ax.set_title('{} Radiohalo'.format(decayList[0][0]))
    return ax

def slicePlot():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal')
    slice_colors = point_colors[np.abs(points[:,2])- flakeNumber*biotiteFlakeThickness < biotiteFlakeThickness] if colorify else 'k'
    ax.scatter(slice[:,0], slice[:,1], s=1, c=slice_colors, yunits='meters', xunits='meters')
    ax.set_title('{} Radiohalo Slice (flake {})'.format(decayList[0][0], flakeNumber))

    crystal_patch = plt.Polygon([[0.e-6, 1.e-6], [-0.3e-6, 0.7e-6], [-0.3e-6, -0.7e-6], [0.e-6, -1.e-6], [0.3e-6, -0.7e-6], [0.3e-6, 0.7e-6]], edgecolor='k', facecolor='w', linewidth=2)
    
    if flakeNumber == 0:
        ax.add_patch(crystal_patch)
    return ax

def damageByRadiusPlot():

    r = np.linspace(np.max(energies_normalized), 0, 300, endpoint=False, dtype=np.longdouble) # Radii included in halo, exclude 0, include maxHaloRadius

    if alternateDecays:
        energy_list = np.concatenate((energies_normalized, alt_energies_normalized))
        weights = np.concatenate((np.ones_like(ratios) - ratios, ratios))

    else:
        energy_list = energies_normalized
        weights = np.ones_like(energies_normalized)

    multiplier = np.sum((np.broadcast_to(energy_list, (len(r), len(energy_list))).T > r).T * weights , axis=-1) 
    singleDecayDamage = solidAngle(r, alphaRadius)
    damageArea = multiplier * singleDecayDamage
    correctedDamageArea = damageArea[damageArea > 0]
    # capHeight = damageArea * r/tau
    
    nonOverlappingCalibrationCurve = np.divide(np.min(correctedDamageArea), correctedDamageArea)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(r[damageArea > 0], correctedDamageArea, s=2)
    ax.set_yscale('log')
    ax.set_title('Damage factor vs radius from center ({})'.format(decayList[0][0]))
    ax.set_ylabel('Single nucleon chain decay area [sr]')
    return ax

def thresholdAgePlot(threshold):
    r = np.linspace(np.max(energies_normalized), 0, 300, endpoint=False, dtype=np.longdouble) # Radii included in halo, exclude 0, include maxHaloRadius

    if alternateDecays:
        energy_list = np.concatenate((energies_normalized, alt_energies_normalized))
        weights = np.concatenate((np.ones_like(ratios) - ratios, ratios))

    else:
        energy_list = energies_normalized
        weights = np.ones_like(energies_normalized)

    multiplier = np.sum((np.broadcast_to(energy_list, (len(r), len(energy_list))).T > r).T * weights , axis=-1) 
    shell_area = 2*tau*np.square(r)

    counts = threshold * (shell_area[multiplier > 0]/multiplier[multiplier > 0])
    # age = counts/(mean life * number of P atoms)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(r[multiplier > 0], counts, s=2)
    ax.set_yscale('log')
    ax.set_title('Decays vs visibility radius ({})'.format(decayList[0][0]))
    ax.set_ylabel('Decay count [a]')
    return ax

# spherePlot()
# slicePlot()
# damageByRadiusPlot()
thresholdAgePlot(visibility_dose)

end_time = time.time()
print('End: {}\nRuntime: {}'.format(time.asctime(time.localtime(end_time)), end_time - start_time))

plt.show()

