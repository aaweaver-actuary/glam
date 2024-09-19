"""Utility functions for the glam_plots package."""


def get_marker_props(color: str, opacity: float = 0.5, size: int = 10) -> dict:
    """Return a dictionary with marker properties."""
    return {
        "size": size,
        "color": color,
        "opacity": opacity,
        "line": {"color": "black", "width": 0.5},
    }


def get_line_props(color: str, width: float = 1.5, dash: str = "solid") -> dict:
    """Return a dictionary with line properties."""
    return {"color": color, "width": width, "dash": dash}
