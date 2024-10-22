from enum import Enum

class FieldStatIndex(Enum):
    LocationX = 0,
    VelocityX = 1,
    ForceX = 2,  # force in the direction of positive x
    OffsetX = 3,  # distance between current one and the next by index in its axis line
    #LocationY = 0,
    #LocationZ = 2,

class OutputStatIndex(Enum):
    Amplitude = 0,
    Force = 1  # currently present in FieldStatIndex instead, so it will be empty here
    Frequency = 2,  # not supported yet, relevant for forced

class SideType(Enum):
    x = 0,
    y = 1,
    z = 2
