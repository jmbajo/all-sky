# -*- coding: utf-8 -*-

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from math import radians, degrees
import matplotlib.pyplot as plt
import numpy as np
import datetime

from solver import solve, compute_azimuth_zenith, transform_a_z_to_img, orient2D, \
    tri_area, transform_image, raster_triangle

# Image filename to calibrate
image_data = np.flipud(fits.getdata('./ASICAP_2019-08-23_21_39_22_281.FIT', ext=0))


# List of identified stars in the image
star_list = list()
#                    image coord            azimituh                        zenith
star_list.append( ( ( 669, 1161),  (radians(0.723345280543576),  radians(50.0346078298743)) ) ) # Polaris
star_list.append( ( (2706, 1464),  (radians(211.09338022019),    radians(73.7575296749023)) ) ) # Antares
star_list.append( ( (1335, 1240),  (radians(319.49546140679),    radians(17.4450194260354)) ) ) # Rastaban
star_list.append( ( (1349, 1165),  (radians(327.374758099071),   radians(13.912032284885 )) ) ) # Etamin
star_list.append( ( (1568, 1009),  (radians(224.290804317375),   radians(2.10224265259195)) ) ) # Vega
star_list.append( ( (2028, 593),   (radians(150.204413070105),   radians(34.7206621992614)) ) ) # Altair
star_list.append( ( (1317, 653),   (radians(67.0194831559618),   radians(21.895589068438 )) ) ) # Deneb
star_list.append( ( (1000, 774),   (radians(32.7201647640243),   radians(31.9092826802497)) ) ) # Alderamin
star_list.append( ( (1281, 171),   (radians(82.1251771819885),   radians(48.7946839156508)) ) ) # Matar
star_list.append( ( (1250, 98),    (radians(81.579727947881),    radians(53.7896853505364)) ) ) # Scheat


solution = solve(star_list, (image_data.shape[1], image_data.shape[0]))

dst_size = 4000
# x = 2900
# y = 1990
# vertex1_pos = transform_a_z_to_img(compute_azimuth_zenith(x+1, y+1, **solution), dst_size)
# vertex2_pos = transform_a_z_to_img(compute_azimuth_zenith(x,   y+1, **solution), dst_size)
# vertex3_pos = transform_a_z_to_img(compute_azimuth_zenith(x,   y,   **solution), dst_size)
# vertex4_pos = transform_a_z_to_img(compute_azimuth_zenith(x+1, y,   **solution), dst_size)
#
# print(vertex2_pos)
# print(vertex4_pos)
# print(vertex1_pos)


print("Rastering output image...")
# raster_triangle()

out_data = np.zeros((dst_size, dst_size), dtype=np.uint16)
# raster_triangle((vertex2_pos, vertex4_pos, vertex1_pos), (100, 500, 700), out_data)

transform_image(image_data, out_data, solution)

# print(np.min(out_data), np.max(out_data))

currentDT = datetime.datetime.now()
outfile = 'output' + str(currentDT) + '.fits'

hdu = fits.PrimaryHDU(out_data.astype(image_data.dtype))
hdu.writeto(outfile, overwrite=True)






