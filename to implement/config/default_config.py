# config/default_config.py

# TODO: Not currently implemented

from typing import Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class GameMode(Enum):
    """Game mode settings that affect gameplay."""
    TRAINING = "training"
    EVALUATION = "evaluation"
    DEMO = "demo"


class TargetIteration(Enum):
    """Different target count configurations."""
    A = 10  # 10 ships
    B = 20  # 20 ships
    C = 30  # 30 ships
    D = 50  # 50 ships
    E = 100  # 100 ships


class MotionIteration(Enum):
    """Different ship movement patterns."""
    F = 0  # Stationary
    G = 5  # Slow movement
    H = 10  # Medium movement
    I = 15  # Fast movement
    J = "random_slow"  # Random: 50% still, 30% slow, 15% medium, 5% fast
    K = "random_fast"  # Random: 50% fast, 30% medium, 15% slow, 5% still


@dataclass
class DisplayConfig:
    """Display and window configuration."""
    window_size: Tuple[int, int] = (1800, 850)
    gameboard_size: int = 700
    gameboard_border_margin: int = 35
    fps: int = 60
    show_agent_waypoint: int = 2  # 0-3 waypoints shown
    show_agent_location: str = "persistent"  # "persistent", "spotty", "none"


@dataclass
class GameplayConfig:
    """Core gameplay settings."""
    num_aircraft: int = 2
    target_iteration: TargetIteration = TargetIteration.D
    motion_iteration: MotionIteration = MotionIteration.F
    game_duration: int = 120  # seconds
    search_pattern: str = "ladder"
    mode: GameMode = GameMode.TRAINING


@dataclass
class TransparencyConfig:
    """Settings for agent transparency/explanability."""
    show_current_action: bool = True
    show_risk_info: bool = True
    show_decision_rationale: bool = True
    show_priorities: bool = True


@dataclass
class ScoreConfig:
    """Point values for different actions."""
    all_targets_points: int = 20  # All targets identified
    target_points: int = 10  # Each target identified
    threat_points: int = 5  # Each threat identified
    time_points: int = 2  # Points per second remaining
    agent_damage_points: float = -0.1  # Points per damage to AI
    human_damage_points: float = -0.2  # Points per damage to human
    wingman_dead_points: int = -30  # AI aircraft destroyed
    human_dead_points: int = -40  # Human aircraft destroyed


class Colors:
    """Color definitions used throughout the game."""
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 128, 0)
    BLUE = (0, 0, 255)
    GOLD = (255, 215, 0)
    PURPLE = (128, 0, 128)

    # UI Colors
    UI_BACKGROUND = (100, 100, 100)
    UI_PANEL = (230, 230, 230)
    UI_HEADER = (200, 200, 200)

    # Aircraft Colors
    AIRCRAFT_COLORS = [
        (0, 160, 160),  # Teal
        (0, 0, 255),  # Blue
        (200, 0, 200),  # Purple
        (80, 80, 80)  # Gray
    ]

    # Ship Colors
    AGENT_COLOR_UNOBSERVED = (255, 215, 0)  # Gold
    AGENT_COLOR_OBSERVED = (128, 0, 128)  # Purple
    AGENT_COLOR_THREAT = (255, 0, 0)  # Red


class AgentConstants:
    """Constants related to agent rendering and behavior."""
    BASE_DRAW_WIDTH = 10  # Base scale unit for drawing
    THREAT_RADIUS = [0, 1.4, 2.5, 4]  # Radius multiplier for each threat level

    # Aircraft rendering constants
    NOSE_LENGTH = 10  # Pixels forward of wings
    TAIL_LENGTH = 25  # Pixels behind wings
    TAIL_WIDTH = 7  # Pixels perpendicular to body
    WING_LENGTH = 18  # Pixels perpendicular to body
    LINE_WIDTH = 5  # Pixel width of lines

    # Aircraft capability constants
    ENGAGEMENT_RADIUS = 40  # Pixel width for WEZ identification
    ISR_RADIUS = 85  # Pixel width for target identification


class UIConstants:
    """Constants for UI layout and rendering."""
    FONT_SIZES = {
        "small": 24,
        "medium": 30,
        "large": 36,
        "title": 74
    }

    BUTTON_SIZES = {
        "standard": (180, 60),
        "large": (375, 65),
        "square": (100, 100)
    }

    PANEL_SIZES = {
        "gameplan": (405, 555),
        "status": (445, 340),
        "comm_log": (445, 210)
    }

    MAX_COMM_MESSAGES = 7


def load_config() -> Dict[str, Any]:
    """
    Create default configuration dictionary.
    Override values from environment variables or config file if present.
    """
    config = {
        "display": DisplayConfig(),
        "gameplay": GameplayConfig(),
        "transparency": TransparencyConfig(),
        "scoring": ScoreConfig(),
        "colors": Colors(),
        "agent_constants": AgentConstants(),
        "ui_constants": UIConstants(),
    }

    # TODO: Override with environment variables
    # TODO: Override with config file if present
    # TODO: Validate configuration

    return config


# config/constants.py

from typing import Dict, List, Tuple
from enum import Enum


class FlightPatterns:
    """Predefined flight patterns for aircraft."""
    PATTERNS: Dict[str, List[Tuple[float, float]]] = {
        "square": [(0, 1), (0, 0), (1, 0), (1, 1)],
        "ladder": [(1, 1), (1, .66), (0, .66), (0, .33),
                   (1, .33), (1, 0), (0, 0)],
        "hold": [(.4, .4), (.4, .6), (.6, .6), (.6, .4)]
    }

    EDGE_MARGIN = 0.2  # Distance from board edge as proportion


class QuadrantBounds:
    """Quadrant definitions for search patterns."""

    @staticmethod
    def get_bounds(quadrant: str, board_size: int) -> Tuple[float, float, float, float]:
        """Returns (min_x, max_x, min_y, max_y) for given quadrant."""
        half = board_size * 0.5
        bounds = {
            "full": (0, board_size, 0, board_size),
            "NW": (0, half, 0, half),
            "NE": (half, board_size, 0, half),
            "SW": (0, half, half, board_size),
            "SE": (half, board_size, half, board_size)
        }
        return bounds[quadrant]


class EventTypes(Enum):
    """Custom event types used by the game."""
    TARGET_IDENTIFIED = "target_identified"
    THREAT_IDENTIFIED = "threat_identified"
    AGENT_DAMAGED = "agent_damaged"
    AGENT_DESTROYED = "agent_destroyed"
    GAME_COMPLETE = "game_complete"
    POLICY_CHANGED = "policy_changed"