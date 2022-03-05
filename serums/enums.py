"""Enumerations used by SERUMS."""
import enum


class GSMTypes(enum.Enum):
    STUDENTS_T = enum.auto()
    CAUCHY = enum.auto()
    SYMMETRIC_A_STABLE = enum.auto()

    def __str__(self):
        """Return the enum name for strings."""
        return self.name
