import csv
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import AltAz
from astropy.utils import iers

import numpy as np
import matplotlib.pyplot as plot

# TODO ver esto
iers.conf.auto_download = False


class StarsDB:
    def __init__(self, lat, lon, date, time, height=0):
        self.location = {}
        self.datetime = {}

        self.star_list = dict()

        print("Reading star file...")
        #http://www.astronexus.com/hyg
        with open('stars.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            stars_count = len(list(reader))

            #TODO CUIDADO ACA
            # stars_count = 1000

        with open('stars.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            self.star_list["id"] = np.empty(shape=stars_count, dtype=float)
            self.star_list["name"] = list()

            self.star_list["ra"] = np.empty(shape=stars_count, dtype=float)
            self.star_list["dec"] = np.empty(shape=stars_count, dtype=float)

            self.star_list["az"] = np.empty(shape=stars_count, dtype=float)
            self.star_list["z"] = np.empty(shape=stars_count, dtype=float)

            self.star_list["mag"] = np.empty(shape=stars_count, dtype=float)
            self.star_list["absmag"] = np.empty(shape=stars_count, dtype=float)

            i = 0
            for row in reader:
                self.star_list["id"][i] = row["id"]
                self.star_list["name"].append(row["proper"].lower())

                self.star_list["ra"][i] = row["ra"]
                self.star_list["dec"][i] = row["dec"]

                self.star_list["az"][i] = 0.0
                self.star_list["z"][i] = 0.0

                self.star_list["mag"][i] = row["mag"]
                self.star_list["absmag"][i] = row["absmag"]

                i += 1
                # if i == 1000:
                #     break

        print("Reading star file... Done")
        self.update_a_z(lat, lon, date, time, height=0)


    def update_a_z(self, lat, lon, date, time, height=0):
        self.location = EarthLocation(lat=lat, lon=lon, height=height * u.m)
        self.datetime = Time("{} {}".format(date, time))
        conv_obj = AltAz(location=self.location, obstime=self.datetime)

        coord = SkyCoord(ra=self.star_list["ra"],
                         dec=self.star_list["dec"],
                         unit=(u.hourangle, u.deg))

        print("Computing A/Z coordinates...")
        coords = coord.transform_to(conv_obj)
        print("Computing A/Z coordinates... Done")

        print("Moving A/Z coordinates...")
        self.star_list["az"] = coords.az.deg
        self.star_list["z"] = 90.0 - coords.alt.deg
        print("Moving A/Z coordinates... Done")


    def filter(self, names=None, az_range=None, z_range=None, absmag_range=None, mag_range=None):
        if names is not None:
            _names = [name.lower() for name in names]
            name_filter = np.array([name.lower() in _names for name in self.star_list["name"]])
        else:
            name_filter = np.ones(len(self.star_list["name"]))

        if az_range is not None:
            az_range_filter = np.logical_and(az_range[0] < self.star_list["az"], self.star_list["az"] < az_range[1])
        else:
            az_range_filter = np.ones(len(self.star_list["az"]))

        if z_range is not None:
            z_range_filter = np.logical_and(z_range[0] < self.star_list["z"], self.star_list["z"] < z_range[1])
        else:
            z_range_filter = np.ones(len(self.star_list["z"]))

        if absmag_range is not None:
            absmag_range_filter = np.logical_and(absmag_range[0] < self.star_list["absmag"], self.star_list["absmag"] < absmag_range[1])
        else:
            absmag_range_filter = np.ones(len(self.star_list["absmag"]))

        if mag_range is not None:
            mag_range_filter = np.logical_and(mag_range[0] < self.star_list["mag"], self.star_list["mag"] < mag_range[1])
        else:
            mag_range_filter = np.ones(len(self.star_list["mag"]))

        return np.where(
            np.logical_and(
                np.logical_and(az_range_filter, z_range_filter),
                np.logical_and(
                    np.logical_and(absmag_range_filter, name_filter),
                    mag_range_filter))
        )[0]


    # return a list of stars from a list of indices
    def get_stars(self, indices):
        return [self.get_star(i) for i in indices]


    # return a stars from a index
    def get_star(self, index):
        return {"name": self.star_list["name"][index],
                "ra": self.star_list["ra"][index],
                "dec": self.star_list["dec"][index],
                "az": self.star_list["az"][index],
                "z": self.star_list["z"][index]}


# thesis
# s = StarsDB(lat=34.9506, lon=-106.45963, date="2012-05-20", time="04:25:12")

#sharolyn
s = StarsDB(lat=40.325917, lon=-105.598346, date="2019-08-24", time="03:39:00")

#Bahia
# s = StarsDB(lat=-38.715789, lon=-62.263934, date="2019-12-13", time="23:59:00")


# indices = s.filter(mag_range=(-100,3), z_range=(0,90))
indices = s.filter(names=["vega", "altair", "deneb", "alderamin", "polaris", "etamin", "rastaban", "matar", "scheat", "antares"])
for st in s.get_stars(indices):
    print(st)

indices = s.filter(az_range=(0,90), z_range=(0,90))
for st in s.get_stars(indices):
    print(st)

# print(s.get_stars(indices))

# plot.axes(projection='polar')

# Set the title of the polar plot
# plot.title('Estrellas')

# Plot a circle with radius 2 using polar form
# rads = np.arange(0, (2*np.pi), 0.01)

# for radian in rads:
# plot.polar(np.deg2rad(s.star_list["az"][indices]), s.star_list["z"][indices], '.')
# print(len(indices))

    # plot.text(np.deg2rad(st["az"]), st["z"], st["name"])


# Display the Polar plot
plot.show()

