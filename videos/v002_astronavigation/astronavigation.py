import datetime
from math import pi, sin, cos, tan, asin, radians, degrees
import numpy as np
import pandas as pd
import pytz

def datetime_to_radians(time: datetime.datetime) -> float:
    time = time.astimezone(pytz.utc)
    return (
        time.hour * 2 * pi / 24
        + time.minute * 2 * pi / (24 * 60)
        + time.second * 2 * pi / (24 * 60 * 60)
    )

def atmospheric_refraction(altitude):
    """
    Based on:
    https://thenauticalalmanac.com/Formulas.html#Determine_Refraction_
    """
    # return altitude + 7.31 / (altitude + 4.4)
    altitude_degrees = degrees(altitude)
    return radians(
        1
        / (
            tan(radians(altitude_degrees + 7.31 / (altitude_degrees + 4.4)))
            * 60
        )
    )

def datetime_to_juliandate(dt: datetime):
    """
    Convert a datetime object to Julian date.
    """
    dt = dt.astimezone(pytz.utc)

    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    if month <= 2:
        year -= 1
        month += 12

    A = year // 100
    B = 2 - A + A // 4
    C = -0.75
    if year > 0:
        C = 0

    JD = int(365.25 * year + C) + int(30.6001 * (month + 1)) + day + B + 1720994.5

    JD += (hour + (minute / 60.0) + (second / 3600.0)) / 24.0

    return JD

def position_circle(h: float, time: datetime.datetime, declination=None):
    if declination is None:
        declination = sun_declination(time)
    eot = equation_of_time(time)
    points_no = 180 * 60 * 60

    altitudes = np.linspace(-pi / 2, pi / 2, points_no)
    cos_t = np.divide(
        sin(h) - sin(declination) * np.sin(altitudes),
        cos(declination) * np.cos(altitudes),
    )

    df = pd.DataFrame(data={"alt": altitudes, "cos_t": cos_t, "row": range(points_no)})
    df = df[(df["cos_t"] <= 1) & (df["cos_t"] >= -1)]

    df["lambda_1"] = np.arccos(df["cos_t"]) - (eot + datetime_to_radians(time)) + pi
    df["lambda_2"] = -np.arccos(df["cos_t"]) - (eot + datetime_to_radians(time)) + pi

    return df

def sun_declination(timestamp: datetime.datetime) -> float:
    """
    based on: https://aa.usno.navy.mil/faq/sun_approx
    """
    d = datetime_to_juliandate(timestamp) - 2_451_545.0

    g_degrees = 357.529 + 0.98560028 * d
    g = radians(g_degrees)
    q = 280.459 + 0.98564736 * d
    l = q + 1.915 * sin(g) + 0.020 * sin(2 * g)

    return asin(sin(radians(l)) * sin(radians(23.439 - 0.00000036 * d)))

def equation_of_time(timestamp):
    """
    based on
    https://celestialprogramming.com/snippets/equationoftime-simple.html
    """
    jd = datetime_to_juliandate(timestamp)

    t = (jd - 2415020.0) / 36525

    epsilon = radians(
        23.452_294 - 0.0130_125 * t - 0.000_00164 * t**2 + 0.000_000_503 * t**3
    )
    y = tan(epsilon / 2) ** 2

    l = radians(279.69668 + 36_000.76892 * t + 0.000_3025 * t**2)
    e = 0.016_75104 - 0.000_0418 * t - 0.000_000_126 * t**2
    m = radians(358.47583 + 35_999.049_75 * t - 0.00015 * t**2 - 0.000_003_3 * t**3)

    return (
        y * sin(2 * l)
        - 2 * e * sin(m)
        + 4 * e * y * sin(m) * cos(2 * l)
        - 0.5 * y**2 * sin(4 * l)
        - 1.25 * e**2 * sin(2 * m)
    )

def radians_as_degrees(angle):
    dgr = degrees(angle)
    complete_degrees = int(dgr)
    minutes = (dgr - complete_degrees) * 60
    complete_minutes = int(minutes)
    seconds = (minutes - complete_minutes) * 60

    return f"{complete_degrees} {complete_minutes}' {seconds}''"

def get_position_circles_intersection(df1, df2):
    common_alts = df1.merge(df2, how="inner", on="row")

    solutions = []

    solutions += _curves_intersection(
        common_alts["lambda_1_x"], common_alts["lambda_1_y"], common_alts["alt_x"]
    )
    solutions += _curves_intersection(
        common_alts["lambda_1_x"], common_alts["lambda_2_y"], common_alts["alt_x"]
    )
    solutions += _curves_intersection(
        common_alts["lambda_2_x"], common_alts["lambda_1_y"], common_alts["alt_x"]
    )
    solutions += _curves_intersection(
        common_alts["lambda_2_x"], common_alts["lambda_2_y"], common_alts["alt_x"]
    )

    return solutions


def _curves_intersection(y1, y2, x):
    idx = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
    solutions = []
    for ind in idx:
        a = abs(y1[ind] - y2[ind])
        A = abs(y1[ind + 1] - y2[ind + 1]) + a
        b_x = (x[ind + 1] - x[ind]) * a / A
        b_y = (y1[ind + 1] - y1[ind]) * a / A

        solutions.append([x[ind] + b_x, y1[ind] + b_y])
    return solutions


def get_positions(
        h1: float,
        time1: datetime.datetime,
        h2: float,
        time2: datetime.datetime,
        declination1=None,
        declination2=None,
        refraction_correction=True,
    ):
    if refraction_correction:
        h1 -= atmospheric_refraction(h1)
        h2 -= atmospheric_refraction(h2)

    circle1 = position_circle(h1, time1, declination1)
    circle2 = position_circle(h2, time2, declination2)

    return get_position_circles_intersection(circle1, circle2)


def main():
    # altitudes obtained from Stellarium for Krakow
    # N 50 3' 41.15''
    # E 19 56' 11.69
    positions = get_positions(
        h1=radians(50 + 21 / 60),
        time1=datetime.datetime(
            2024,
            4,
            21,
            11,
            43,
            1,
            tzinfo=datetime.timezone(offset=datetime.timedelta(hours=2)),
        ),
        h2=radians(49 + 51 / 60),
        time2=datetime.datetime(
            2024,
            4,
            21,
            13,
            43,
            1,
            tzinfo=datetime.timezone(offset=datetime.timedelta(hours=2)),
        ),
    )

    for position in positions:
        latitude, longitude = position
        n_or_s = "N" if latitude > 0 else "S"
        w_or_e = "E" if longitude > 0 else "W"

        formatted_latitude = radians_as_degrees(abs(latitude))
        formatted_longitude = radians_as_degrees(abs(longitude))
        print(f"{formatted_latitude} {n_or_s} {formatted_longitude} {w_or_e}")


if __name__ == "__main__":
    main()
