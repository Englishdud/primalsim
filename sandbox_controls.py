"""Creator keyboard / mouse controls for the Primal Survival Simulation.

All key bindings are registered on the provided ShowBase app instance.
Free-camera WASD + mouse-look runs as a per-frame task.
"""

import math

from panda3d.core import KeyboardButton, Vec3, WindowProperties

from config import (
    AGENT_COUNT, FREE_CAM_SPEED, PHYSICS_TIMESTEP,
    MAMMOTH_INITIAL_COUNT,
)


class SandboxControls:
    """Registers and handles all creator keyboard/mouse controls.

    The App (play.py's PrimalSimApp) is expected to expose:
        - app.selected_agent (int, read/write)
        - app.hud (HUD instance)
        - app.mammoth_mgr (MammothManager)
        - app.agents (list[Humanoid])
        - app.reset_world() (callable)
        - app.camera (NodePath)
        - app.win (GraphicsWindow)
        - app.mouseWatcherNode (MouseWatcher)
        - app.taskMgr
        - app.accept / app.ignore
    """

    def __init__(self, app):
        """Bind all keys and install the free-camera polling task.

        Args:
            app: The ShowBase subclass from play.py.
        """
        self.app = app
        self.free_cam: bool = False
        self._cam_heading: float = 0.0    # degrees
        self._cam_pitch:   float = -15.0  # degrees (slightly down)
        self._last_mouse_x: float | None = None
        self._last_mouse_y: float | None = None
        self._mouse_sensitivity: float = 0.15   # degrees per pixel
        self._register_keys()
        app.taskMgr.add(self._free_cam_task, 'sandbox_free_cam_task')

    # ------------------------------------------------------------------
    # Key registration
    # ------------------------------------------------------------------

    def _register_keys(self) -> None:
        """Register all keyboard event handlers with Panda3D."""
        self.app.accept('m',      self._spawn_mammoth)
        self.app.accept('h',      self._spawn_herd)
        self.app.accept('r',      self._reset)
        self.app.accept('c',      self._cycle_agent)
        self.app.accept('f',      self._toggle_free_cam)
        self.app.accept('escape', self._quit)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _spawn_mammoth(self) -> None:
        """Spawn a single mammoth at a random sandbox location."""
        self.app.mammoth_mgr.spawn_random()

    def _spawn_herd(self) -> None:
        """Spawn a herd of 2-4 mammoths clustered near a random point."""
        import random
        from config import WORLD_SIZE
        half = WORLD_SIZE / 2 - 50
        hx = random.uniform(-half, half)
        hy = random.uniform(-half, half)
        count = random.randint(2, 4)
        for _ in range(count):
            ox = hx + random.uniform(-20, 20)
            oy = hy + random.uniform(-20, 20)
            self.app.mammoth_mgr.spawn([ox, oy, 0.0])

    def _reset(self) -> None:
        """Trigger a full simulation reset."""
        self.app.reset_world()

    def _cycle_agent(self) -> None:
        """Cycle the follow-camera to the next live agent."""
        for attempt in range(AGENT_COUNT):
            nxt = (self.app.selected_agent + 1 + attempt) % AGENT_COUNT
            if nxt < len(self.app.agents) and not self.app.agents[nxt].is_dead:
                self.app.selected_agent = nxt
                return
        # Fall back to cycling all agents even if dead
        self.app.selected_agent = (self.app.selected_agent + 1) % AGENT_COUNT

    def _toggle_free_cam(self) -> None:
        """Toggle between free-fly camera and follow camera."""
        self.free_cam = not self.free_cam
        if self.free_cam:
            self._capture_mouse()
        else:
            self._release_mouse()

    def _quit(self) -> None:
        """Exit the application."""
        self.app.userExit()

    # ------------------------------------------------------------------
    # Free camera task
    # ------------------------------------------------------------------

    def _free_cam_task(self, task):
        """Poll WASD keys and mouse delta each frame for free-fly camera."""
        if not self.free_cam:
            return task.cont

        mwn = self.app.mouseWatcherNode
        dt  = self.app.taskMgr.globalClock.getDt()

        # Mouse look
        if self.app.win.hasPointer(0):
            pointer = self.app.win.getPointer(0)
            mx = pointer.getX()
            my = pointer.getY()
            cx = self.app.win.getProperties().getXSize() // 2
            cy = self.app.win.getProperties().getYSize() // 2
            if self._last_mouse_x is not None:
                dx = mx - cx
                dy = my - cy
                self._cam_heading -= dx * self._mouse_sensitivity
                self._cam_pitch    = max(-89.0, min(89.0,
                                         self._cam_pitch - dy * self._mouse_sensitivity))
            self.app.win.movePointer(0, cx, cy)
            self._last_mouse_x = cx
            self._last_mouse_y = cy

        self.app.camera.setHpr(self._cam_heading, self._cam_pitch, 0.0)

        # WASD movement along camera local axes
        speed = FREE_CAM_SPEED * dt
        move  = Vec3(0, 0, 0)

        if mwn.isButtonDown(KeyboardButton.asciiKey('w')):
            move.y += speed
        if mwn.isButtonDown(KeyboardButton.asciiKey('s')):
            move.y -= speed
        if mwn.isButtonDown(KeyboardButton.asciiKey('a')):
            move.x -= speed
        if mwn.isButtonDown(KeyboardButton.asciiKey('d')):
            move.x += speed
        if mwn.isButtonDown(KeyboardButton.asciiKey('q')):
            move.z -= speed
        if mwn.isButtonDown(KeyboardButton.asciiKey('e')):
            move.z += speed

        if move.lengthSquared() > 0:
            self.app.camera.setPos(
                self.app.camera,   # relative to own frame
                move,
            )

        return task.cont

    # ------------------------------------------------------------------
    # Mouse capture helpers
    # ------------------------------------------------------------------

    def _capture_mouse(self) -> None:
        """Hide and centre the mouse cursor for free-look mode."""
        props = WindowProperties()
        props.setCursorHidden(True)
        self.app.win.requestProperties(props)
        cx = self.app.win.getProperties().getXSize() // 2
        cy = self.app.win.getProperties().getYSize() // 2
        self.app.win.movePointer(0, cx, cy)
        self._last_mouse_x = None
        self._last_mouse_y = None

    def _release_mouse(self) -> None:
        """Restore the mouse cursor after leaving free-look mode."""
        props = WindowProperties()
        props.setCursorHidden(False)
        self.app.win.requestProperties(props)
        self._last_mouse_x = None
        self._last_mouse_y = None
