"""Panda3D OnscreenText HUD overlay for the Primal Survival Simulation.

Displays per-agent status panels (hunger, stamina, status label), global
simulation info, and the key-binding reference strip at the bottom of the
screen.
"""

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

from config import AGENT_COUNT, AGENT_COLORS, HUNGER_MAX, STAMINA_MAX

# Unicode block characters for progress bars
_FULL_BLOCK  = '\u2588'
_EMPTY_BLOCK = '\u2591'
_BAR_WIDTH   = 10   # characters wide


def _make_bar(value: float, max_value: float) -> str:
    """Return a unicode block bar string representing value/max_value."""
    frac   = max(0.0, min(1.0, value / max(max_value, 1e-6)))
    filled = round(frac * _BAR_WIDTH)
    empty  = _BAR_WIDTH - filled
    return _FULL_BLOCK * filled + _EMPTY_BLOCK * empty


def _rgba_to_panda(rgba: tuple) -> tuple:
    """Return (r, g, b, a) for Panda3D OnscreenText fg parameter."""
    return rgba  # already float 0-1


# Panel positions in 2-D render space (aspect2d).
# Panda3D's default aspect2d: x ∈ [-aspect, +aspect], y ∈ [-1, +1]
# For a 16:9 display aspect ≈ 1.78.  We use -1.3 / 0.6 for left/right columns.
_PANEL_POSITIONS = [
    (-1.30,  0.90),   # Agent 0 – top-left
    ( 0.60,  0.90),   # Agent 1 – top-right
    (-1.30, -0.50),   # Agent 2 – bottom-left
    ( 0.60, -0.50),   # Agent 3 – bottom-right
]

_PANEL_SCALE   = 0.043
_SHADOW_OFFSET = (0.003, -0.003)


class HUD:
    """Manages all OnscreenText elements for the simulation HUD."""

    def __init__(self, base_app):
        """Create all text nodes.

        Args:
            base_app: The ShowBase instance from play.py.
        """
        self.app = base_app
        self._visible: bool = True

        # Per-agent text nodes
        self._agent_texts: list[OnscreenText] = []
        for i in range(AGENT_COUNT):
            x, y = _PANEL_POSITIONS[i]
            color = _rgba_to_panda(AGENT_COLORS[i])
            text = OnscreenText(
                text='',
                pos=(x, y),
                scale=_PANEL_SCALE,
                fg=color,
                shadow=(0.0, 0.0, 0.0, 0.9),
                shadowOffset=_SHADOW_OFFSET,
                align=TextNode.ALeft,
                mayChange=True,
            )
            self._agent_texts.append(text)

        # Global info panel – top-centre
        self._info_text = OnscreenText(
            text='',
            pos=(0.0, 0.93),
            scale=_PANEL_SCALE,
            fg=(1.0, 1.0, 1.0, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.9),
            shadowOffset=_SHADOW_OFFSET,
            align=TextNode.ACenter,
            mayChange=True,
        )

        # Controls strip – bottom-centre
        self._controls_text = OnscreenText(
            text=(
                'M: Spawn Mammoth  |  H: Spawn Herd  |  R: Reset  '
                '|  C: Cycle Camera  |  F: Free Cam  |  ESC: Quit'
            ),
            pos=(0.0, -0.95),
            scale=0.035,
            fg=(0.9, 0.9, 0.9, 1.0),
            shadow=(0.0, 0.0, 0.0, 0.8),
            shadowOffset=_SHADOW_OFFSET,
            align=TextNode.ACenter,
            mayChange=False,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        agent_stats: list[dict],
        selected_agent: int,
        sim_time: float,
        mammoths_alive: int,
        agents_alive: int,
    ) -> None:
        """Refresh all HUD text with current simulation state.

        Args:
            agent_stats:     List of dicts, one per agent, with keys:
                             hunger, stamina, is_holding_tool, is_dead,
                             status_label (str).
            selected_agent:  Index of camera-followed agent (highlighted).
            sim_time:        Elapsed simulation seconds.
            mammoths_alive:  Current live mammoth count.
            agents_alive:    Current live agent count.
        """
        if not self._visible:
            return

        for i, stats in enumerate(agent_stats):
            hunger   = float(stats.get('hunger',  100.0))
            stamina  = float(stats.get('stamina', 100.0))
            is_dead  = bool(stats.get('is_dead',  False))
            has_tool = bool(stats.get('is_holding_tool', False))
            label    = str(stats.get('status_label', 'IDLE'))

            hunger_bar  = _make_bar(hunger,  HUNGER_MAX)
            stamina_bar = _make_bar(stamina, STAMINA_MAX)

            selected_mark = ' ◄' if i == selected_agent else ''
            tool_str      = 'SPEAR' if has_tool else '─────'

            lines = [
                f'Agent {i+1}{selected_mark}',
                f'Hunger : [{hunger_bar}] {hunger:5.1f}%',
                f'Stamina: [{stamina_bar}] {stamina:5.1f}%',
                f'Tool   : {tool_str}',
                f'Status : {label}',
            ]
            self._agent_texts[i].setText('\n'.join(lines))

        # Global info
        mins = int(sim_time) // 60
        secs = int(sim_time) % 60
        self._info_text.setText(
            f'Time: {mins:02d}:{secs:02d}  |  '
            f'Mammoths: {mammoths_alive}  |  '
            f'Agents: {agents_alive}/{AGENT_COUNT}'
        )

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        """Show or hide the entire HUD."""
        self._visible = visible
        for text in self._agent_texts:
            text.show() if visible else text.hide()
        self._info_text.show() if visible else self._info_text.hide()
        self._controls_text.show() if visible else self._controls_text.hide()

    def toggle_visible(self) -> None:
        """Flip HUD visibility."""
        self.set_visible(not self._visible)

    def destroy(self) -> None:
        """Clean up all OnscreenText nodes."""
        for text in self._agent_texts:
            text.destroy()
        self._info_text.destroy()
        self._controls_text.destroy()
