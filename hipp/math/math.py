import math
"""
Library for basic math operations.
"""
    
def midpoint(x1, y1, x2, y2):
    """
    Computes midpoint between two points.
    """
    return ((y1 + y2)/2,(x1 + x2)/2) 
    
def slope(x1, y1, x2, y2):
    """
    Computes slope of line connecting two points.
    """
    return (y2-y1)/(x2-x1)
    
def intersection_angle(m1, m2):
    """
    Computes intersection angle between two slopes.
    """
    return math.degrees(math.atan((m2-m1) / (1+m1*m2)))
    
def distance(p1,p2):
    """
    Computes distance between two points.
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) 
