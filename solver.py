import numpy as np
from scipy import optimize
from math import atan2, cos, sin, radians, acos, degrees, fmod, pi, asin


# Thesis Eq. 6
def r(x, x0, y, y0):
    return pow(((x - x0)*(x - x0) + (y - y0)*(y - y0)), 0.5)


# Thesis Eq. 7
def u(P1, P2, P3, x, x0, y, y0):
    return P1 * r(x, x0, y, y0)*r(x, x0, y, y0) + P2 * r(x, x0, y, y0) + P3


# Thesis Eq. 8
def b(a0, E, x, x0, y, y0):
    return a0 - E + atan2(y - y0, x - x0)


# Thesis Eq. 9  // Computes de zenith
def f_cos(x, y, a, z, x0, y0, P1, P2, P3, a0, E, eps):
    u_ = u(P1, P2, P3, x, x0, y, y0)
    b_ = b(a0, E, x, x0, y, y0)

    return cos(z) - cos(u_) * cos(eps) + sin(u_) * sin(eps) * cos(b_)

# Thesis Eq. 10 // Computes the azimuth
def f_sin(x, y, a, z, x0, y0, P1, P2, P3, a0, E, eps):
    u_ = u(P1, P2, P3, x, x0, y, y0)
    b_ = b(a0, E, x, x0, y, y0)

    return (sin(b_) * sin(u_)) / sin(z) - sin(a-E)


# X = unknows_tuple (x0, y0, P1, P2, P3, a0, E, eps)
# stars = List of star tuples (x, y, a, z)
def f_total(X, stars):
    out_list = []
    for star in stars:
        x, y = star[0]
        a, z = star[1]

        out_list.append(f_cos(x, y, a, z, *X))
        out_list.append(f_sin(x, y, a, z, *X))

    return out_list


def compute_azimuth_zenith(x, y, x0, y0, P1, P2, P3, a0, E, eps):
    u_ = u(P1, P2, P3, x, x0, y, y0)
    b_ = b(a0, E, x, x0, y, y0)
    z_solved = acos(cos(u_) * cos(eps) - sin(u_) * sin(eps) * cos(b_))

    sina_E = (sin(b_) * sin(u_)) / sin(z_solved)
    a_E = asin(sina_E)
    cosa_E = (cos(u_) - cos(eps) * cos(z_solved)) / (sin(eps) * sin(z_solved))

    if cosa_E < 0:
        a_E = pi - a_E
    a_solved = a_E + E

    if a_solved<0 :
        a_solved = a_solved + 2 * pi

    return a_solved, z_solved


def transform_a_z_to_img(az, out_size):
    return -cos(az[0]) * az[1] / radians(90) * out_size/2.0 + out_size/2.0, \
           -sin(az[0]) * az[1] / radians(90) * out_size/2.0 + out_size/2.0


def bbox(triangle):
    return \
        (min(triangle[0][0], triangle[1][0], triangle[2][0]), max(triangle[0][0], triangle[1][0], triangle[2][0])), \
        (min(triangle[0][1], triangle[1][1], triangle[2][1]), max(triangle[0][1], triangle[1][1], triangle[2][1]))


def orient2D(A, B, point):
    return (B[0]-A[0])*(point[1]-A[1])-(B[1]-A[1])*(point[0]-A[0])


def tri_area(A, B, C):
    return 0.5 * ( (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1]) * (C[0]-A[0]) )


def raster_triangle(triangle_pos, triangle_values, dst):
    dst_size = dst.shape[0]

    v0 = triangle_pos[0]
    v1 = triangle_pos[1]
    v2 = triangle_pos[2]

    # print("Rasterizing ", v0, v1, v2)

    if (0 <= v0[0] < dst_size) and (0 <= v0[1] < dst_size) and \
       (0 <= v1[0] < dst_size) and (0 <= v1[1] < dst_size) and \
       (0 <= v2[0] < dst_size) and (0 <= v2[1] < dst_size) :
            bounding = bbox(triangle_pos)
            # print(bounding)
            for x in range(int(bounding[0][0]), int(bounding[0][1]) + 1):
                for y in range(int(bounding[1][0]), int(bounding[1][1]) + 1):
                    f12 = orient2D(v1, v2, (x, y))
                    f20 = orient2D(v2, v0, (x, y))
                    f01 = orient2D(v0, v1, (x, y))

                    # print("testeo ", x, y, f12, f20, f01)
                    if f12 > 0 and f20 > 0 and f01 > 0:
                        #estoy dentro del triangulo
                        lamda0 = f12 / (f12 + f20 + f01)
                        lamda1 = f20 / (f12 + f20 + f01)
                        lamda2 = f01 / (f12 + f20 + f01)

                        dst[y][x] = lamda0 * triangle_values[0] + \
                                    lamda1 * triangle_values[1] + \
                                    lamda2 * triangle_values[2]

                        # print("     Estoy adentro ", x, y, dst[y][x])
                    # else:
                        # print("     afuera", f12, f20, f01)

    # else:
    #     print("     afuera img ", v0, v1, v2)



def transform_image(source, dst, solution):
    x_size = source.shape[1]
    y_size = source.shape[0]

    dst_size = dst.shape[0] # se asume imagen salida cuadrada

    for x in range(x_size-1):
        print (x)
        for y in range(y_size-1):
            # rasterizo 2 tris

            vertex1_pos = transform_a_z_to_img(compute_azimuth_zenith(x+1, y,   **solution), dst_size)
            vertex2_pos = transform_a_z_to_img(compute_azimuth_zenith(x,   y,   **solution), dst_size)
            vertex3_pos = transform_a_z_to_img(compute_azimuth_zenith(x,   y+1, **solution), dst_size)
            vertex4_pos = transform_a_z_to_img(compute_azimuth_zenith(x+1, y+1, **solution), dst_size)

            vertex1_value = source[y][x+1]
            vertex2_value = source[y][x]
            vertex3_value = source[y+1][x]
            vertex4_value = source[y+1][x+1]

            tri_1_transf = ( vertex3_pos, vertex2_pos, vertex1_pos )
            tri_1_values = ( vertex3_value, vertex2_value, vertex1_value )

            tri_2_transf = ( vertex1_pos, vertex4_pos, vertex3_pos )
            tri_2_values = ( vertex1_value, vertex4_value, vertex3_value )

            raster_triangle(tri_1_transf, tri_1_values, dst)
            raster_triangle(tri_2_transf, tri_2_values, dst)


def solve(star_list, img_size):
    array = np.array([img_size[0]/2.0, img_size[1]/2.0, 0, 0, 0, 0, 0, 0])

    sol = optimize.root(f_total, array, method='lm', args=star_list, tol=0.00000001)
    # sol = optimize.root(f_total, [320, 240, 0, 0, 0, 0, 0, 0], method='lm', args=star_list)
    print(type(sol.x))


    x0, y0, P1, P2, P3, a0, E, eps = sol.x
    eps = fmod(eps, 2 * pi)

    print ("CHECKS!!")
    for star in star_list:
        print("Error", degrees(compute_azimuth_zenith(star[0][0], star[0][1], x0, y0, P1, P2, P3, a0, E, eps)[0]) - degrees(star[1][0]))

    print("\n")
    for star in star_list:
        print("Error", degrees(compute_azimuth_zenith(star[0][0], star[0][1], x0, y0, P1, P2, P3, a0, E, eps)[1]) - degrees(star[1][1]))

    solution = {
        "x0": x0,
        "y0": y0,
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "a0": a0,
        "E" : E,
        "eps": eps
    }

    return solution


# # sol = optimize.root(poly, [4], method='lm', args=(1, -6, 7))
# star_list = list()
#
# # star_list.append((x, y, radians(a), radians(z)))
# # star_list.append((490, 66,  radians(-322+360), radians(82)))
# # star_list.append((355, 359, radians(-175+360), radians(47)))
# # star_list.append((294, 302, radians(-134+360), radians(22)))
# # star_list.append((137, 227, radians(-59+360),  radians(63)))
# # star_list.append((440, 254, radians(-252+360), radians(48)))
# #
# # star_list.append((399, 338, radians(-195+360), radians(48)))
# # star_list.append((334, 287, radians(-176+360), radians(16)))
# # star_list.append((165, 243, radians(-64+360),  radians(52)))
# # star_list.append((471, 223, radians(-263+360), radians(60)))
# # star_list.append((294, 438, radians(-147+360) , radians(70)))
# #
# # star_list.append((439, 313, radians(-213+360), radians(53)))
# # star_list.append((372, 267, radians(-221+360), radians(21)))
# # star_list.append((197, 251, radians(-68+360),  radians(41)))
# # star_list.append((338, 424, radians(-160+360), radians(65)))
# # star_list.append((147, 338, radians(-94+360),  radians(69)))
# # star_list.append((145, 206, radians(-52+360),  radians(62)))
# # star_list.append((496, 188, radians(-271+360), radians(72)))
#
# # star_list.append((490, 66,  radians(322), radians(82)))
# # star_list.append((355, 359, radians(175), radians(47)))
# # star_list.append((294, 302, radians(134), radians(22)))
# # star_list.append((137, 227, radians(59),  radians(63)))
# # star_list.append((440, 254, radians(252), radians(48)))
# #
# # star_list.append((399, 338, radians(195), radians(48)))
# # star_list.append((334, 287, radians(176), radians(16)))
# # star_list.append((165, 243, radians(64),  radians(52)))
# # star_list.append((471, 223, radians(263), radians(60)))
# # star_list.append((294, 438, radians(147), radians(70)))
# #
# # star_list.append((439, 313, radians(213), radians(53)))
# # star_list.append((372, 267, radians(221), radians(21)))
# # star_list.append((197, 251, radians(68),  radians(41)))
# # star_list.append((338, 424, radians(160), radians(65)))
# # star_list.append((147, 338, radians(94),  radians(69)))
# # star_list.append((145, 206, radians(52),  radians(62)))
# # star_list.append((496, 188, radians(271), radians(72)))
#
# # origin at the center
# # star_list.append((878, 121,   radians(0.723345280543576),radians(50.0346078298743)))
# # star_list.append((-1160, 424, radians(211.09338022019),  radians(73.7575296749023)))
# # star_list.append((213, 200,   radians(319.49546140679),  radians(17.4450194260354)))
# # star_list.append((198, 125,   radians(327.374758099071), radians(13.912032284885)))
# # star_list.append((-21, -30,   radians(224.290804317375), radians(2.10224265259195)))
# # star_list.append((-481, -447, radians(150.204413070105), radians(34.7206621992614)))
# # star_list.append((230, -387,  radians(67.0194831559618), radians(21.895589068438)))
# # star_list.append((546, -266,  radians(32.7201647640243), radians(31.9092826802497)))
# # star_list.append((267, -869,  radians(82.1251771819885), radians(48.7946839156508)))
# # star_list.append((297, -942,  radians(81.579727947881),  radians(53.7896853505364)))
#
# # origin at the top left
# star_list.append( ( (669, 1161),  (radians(0.723345280543576), radians(50.0346078298743)) ) )
# star_list.append( ( (2706, 1464), (radians(211.09338022019),   radians(73.7575296749023)) ) )
# star_list.append( ( (1335, 1240), (radians(319.49546140679),   radians(17.4450194260354)) ) )
# star_list.append( ( (1349, 1165), (radians(327.374758099071),  radians(13.912032284885 )) ) )
# star_list.append( ( (1527, 1010), (radians(224.290804317375),  radians(2.10224265259195)) ) )
# star_list.append( ( (2028, 593),  (radians(150.204413070105),  radians(34.7206621992614)) ) )
# star_list.append( ( (1317, 653),  (radians(67.0194831559618),  radians(21.895589068438 )) ) )
# star_list.append( ( (1000, 774),  (radians(32.7201647640243),  radians(31.9092826802497)) ) )
# star_list.append( ( (1281, 171),  (radians(82.1251771819885),  radians(48.7946839156508)) ) )
# star_list.append( ( (1250, 98),   (radians(81.579727947881),   radians(53.7896853505364)) ) )
#
# array = np.array([1548, 1040, 0, 0, 0, 0, 0, 0])
#
# sol = optimize.root(f_total, array, method='lm', args=star_list, tol=0.0000000001)
# # sol = optimize.root(f_total, [320, 240, 0, 0, 0, 0, 0, 0], method='lm', args=star_list)
# print(type(sol.x))
#
# x0, y0, P1, P2, P3, a0, E, eps = sol.x
# eps = fmod(eps, 2*pi)
#
# print("x0= "  + str(x0))
# print("y0= "  + str(y0))
# print("P1= "  + str(P1))
# print("P2= "  + str(P2))
# print("P3= "  + str(P3))
# print("a0= "  + str(a0))
# print("E= "   + str(E))
# print("eps= " + str(eps))
#
# print ("CHECKS!!")
# for star in star_list:
#     print("Error", degrees(compute_azimuth_zenith(star[0][0], star[0][1], x0, y0, P1, P2, P3, a0, E, eps)[0]) - degrees(star[1][0]))
#
# print("\n")
# for star in star_list:
#     print("Error", degrees(compute_azimuth_zenith(star[0][0], star[0][1], x0, y0, P1, P2, P3, a0, E, eps)[1]) - degrees(star[1][1]))
#
#
# # def f_inverse(X, a, z):
# #     x0, y0, P1, P2, P3, a0, E, eps = sol.x
# #
# #     x = X[0]
# #     y = X[1]
# #
# #     u_ = u(P1, P2, P3, x, x0, y, y0)
# #     b_ = b(a0, E, x, x0, y, y0)
# #
# #     return [cos(z) - cos(u_) * cos(eps) + sin(u_) * sin(eps) * cos(b_),
# #             (sin(b_) * sin(u_)) / sin(z) - sin(a - E)]
# #
# # bounds = np.array([(0, 0), (1548*2, 1040*2)])
# # X0_ = np.array([2706, 1464])
# # az = (radians(211.09338022019), radians(73.7575296749023))
# #
# # # star_list.append((2706, 1464, radians(211.09338022019),  radians(73.7575296749023)))
# # # print(degrees(compute_azimuth_zenith(2706.65552163, 1464.65984582, x0, y0, P1, P2, P3, a0, E, eps)[0]))
# # # print(degrees(compute_azimuth_zenith(2706.65552163, 1464.65984582, x0, y0, P1, P2, P3, a0, E, eps)[1]))
# #
# # # sol_z= optimize.root(f_inverse, x0=X0_, method='lm', args=az, tol=0.000001)
# # # print(fmod(sol_z.x[0], 1548*2))
# # # print(fmod(sol_z.x[1], 1040*2))
# #
# # res = optimize.least_squares(f_inverse, x0=X0_, bounds = bounds, args=az, xtol=0.000000001)
# # print(res)

