import collections
import math

# Define the x y point for pixels
Point = collections.namedtuple("Point", ["x", "y"])
Axial_Point = collections.namedtuple("Axial_Point", ['q', 'r'])

_Hex = collections.namedtuple("Hex", ["q", "r", "s"])


def Hex(q, r, s):
    assert not (round(q + r + s) != 0), "q + r + s must be 0"
    return _Hex(q, r, s)


# Add function to create Hex from Axial Coordinates and an axial point as input
def Axial_Hex(axialPoint):
    s = -axialPoint.q - axialPoint.r
    assert not (round(axialPoint.q + axialPoint.r + s) != 0), "q + r + s must be 0"
    return _Hex(axialPoint.q, axialPoint.r, s)

# Specify Orientation and layout for Hex <-> Pixel conversion
Orientation = collections.namedtuple("Orientation", ["f0", "f1", "f2", "f3", "b0", "b1", "b2", "b3", "start_angle"])

layout_pointy = Orientation(math.sqrt(3.0), math.sqrt(3.0) / 2.0, 0.0, 3.0 / 2.0, math.sqrt(3.0) / 3.0, -1.0 / 3.0, 0.0,
                            2.0 / 3.0, 0.5)
layout_flat = Orientation(3.0 / 2.0, 0.0, math.sqrt(3.0) / 2.0, math.sqrt(3.0), 2.0 / 3.0, 0.0, -1.0 / 3.0,
                          math.sqrt(3.0) / 3.0, 0.0)

# Layout has the orientation, size and origin
Layout = collections.namedtuple("Layout", ["orientation", "size", "origin"])


# Function to covert axial hex coordinates to pixel
def hex_to_pixel(layout, h):
    M = layout.orientation
    size = layout.size
    origin = layout.origin
    x = (M.f0 * h.q + M.f1 * h.r) * size.x
    y = (M.f2 * h.q + M.f3 * h.r) * size.y
    return Point(x + origin.x, y + origin.y)


# Function to convert pixel coordinates to Hex
def pixel_to_hex(layout, p):
    M = layout.orientation
    size = layout.size
    origin = layout.origin
    pt = Point((p.x - origin.x) / size.x, (p.y - origin.y) / size.y)
    q = M.b0 * pt.x + M.b1 * pt.y
    r = M.b2 * pt.x + M.b3 * pt.y
    return Hex(q, r, -q - r)


# Get the corner offset depending on the layout (using start angle)
def hex_corner_offset(layout, corner):
    M = layout.orientation
    size = layout.size
    angle = 2.0 * math.pi * (M.start_angle - corner) / 6.0
    return Point(size.x * math.cos(angle), size.y * math.sin(angle))


# Get the corners of the Polygon in pixel coordinates
def polygon_corners(layout, h):
    corners = []
    center = hex_to_pixel(layout, h)
    for i in range(0, 6):
        offset = hex_corner_offset(layout, i)
        corners.append(Point(round(center.x + offset.x, 2), round(center.y + offset.y, 2)))
    return corners
