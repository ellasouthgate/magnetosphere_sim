import tkinter as tk
from collections import deque

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator

# --------------------------------------------------------------------------
# Physics constants and set-up
# --------------------------------------------------------------------------
k = 1e-7
q = 1.602176634e-19
m_e = 9.1093837015e-31
m_p = 1.67262192369e-27
R_E = 6.371e6
M_dipole = 8.0e22
tilt = np.radians(23.4)

dt = 1.0e-9              
STEPS_PER_FRAME = 50     
SAMPLE_EVERY = 5        
TRAIL_SAMPLES = 2500     

M_dipole_vec = np.array([-M_dipole * np.sin(tilt), 0.0, -M_dipole * np.cos(tilt)])

start_altitude = 0.0
start_x = start_altitude * np.sin(tilt)
start_z = R_E + start_altitude * np.cos(tilt)
initial_position = np.array([start_x, 0.0, start_z])


def dipole_field(position):
    radius = np.linalg.norm(position)
    dot_product = np.dot(M_dipole_vec, position)
    return k * (3.0 * position * dot_product - M_dipole_vec * radius**2) / radius**5


def calculate_velocity_matrix(active_mass, is_proton=False):
    velocity_matrix = {}

    
    magnetic = dipole_field(initial_position)
    field_direction = magnetic / np.linalg.norm(magnetic)
    points_outward = np.dot(field_direction, initial_position) > 0.0
    if points_outward != is_proton:
        field_direction = -field_direction

    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(field_direction, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    perpendicular = np.cross(field_direction, reference)
    perpendicular /= np.linalg.norm(perpendicular)

    for energy_kev in (1.0, 3.0, 5.0):
        speed = np.sqrt(2.0 * energy_kev * 1000.0 * q / active_mass)
        velocity_matrix[energy_kev] = {}
        for pitch_deg in (30.0, 45.0, 75.0):
            alpha = np.radians(pitch_deg)
            along = speed * np.cos(alpha) * field_direction
            across = speed * np.sin(alpha) * perpendicular
            velocity_matrix[energy_kev][pitch_deg] = along + across
    return velocity_matrix


v_p = calculate_velocity_matrix(m_e)


def get_B_field(position, storm_mode=False):
    position = np.asarray(position, dtype=float)
    if np.linalg.norm(position) == 0.0:
        return np.zeros(3)

    magnetic = dipole_field(position)
    if storm_mode:
        magnetic[2] -= 0.6e-6
    return magnetic


def calculate_larmor_radius(velocity, magnetic, mass=m_e):
    field_magnitude = np.linalg.norm(magnetic)
    if field_magnitude == 0.0:
        return 0.0
    field_direction = magnetic / field_magnitude
    v_perp = velocity - np.dot(velocity, field_direction) * field_direction
    return mass * np.linalg.norm(v_perp) / (q * field_magnitude)


def calculate_physics_step(position, velocity, storm_mode=False, is_proton=False):
    magnetic = get_B_field(position, storm_mode)
    field_magnitude = np.linalg.norm(magnetic)
    mass = m_p if is_proton else m_e
    charge = q if is_proton else -q
    acceleration = (charge / mass) * np.cross(velocity, magnetic)

    current = q**2 * field_magnitude / (2.0 * np.pi * mass)
    larmor_radius = calculate_larmor_radius(velocity, magnetic, mass)
    if larmor_radius > 0.0:
        induced_field = (4.0 * np.pi * k) * current / (2.0 * larmor_radius)
    else:
        induced_field = 0.0
    return acceleration, current, induced_field, larmor_radius


def euler_step(position, velocity, storm_mode=False, is_proton=False):
    
    magnetic = get_B_field(position, storm_mode)
    field_direction = magnetic / np.linalg.norm(magnetic)
    mass = m_p if is_proton else m_e
    charge = q if is_proton else -q

    acceleration = (charge / mass) * np.cross(velocity, magnetic)

    v_parallel = np.dot(velocity, field_direction) * field_direction
    v_perp = velocity - v_parallel

    new_v_perp = v_perp + acceleration * dt
    new_v_perp *= np.linalg.norm(v_perp) / np.linalg.norm(new_v_perp)

    new_position = position + velocity * dt
    new_velocity = v_parallel + new_v_perp
    return new_position, new_velocity


def get_guiding_center(position, velocity, storm_mode=False, is_proton=False):
    magnetic = get_B_field(position, storm_mode)
    field_magnitude = np.linalg.norm(magnetic)
    if field_magnitude == 0.0:
        return position.copy()
    mass = m_p if is_proton else m_e
    charge = q if is_proton else -q
    direction = magnetic / field_magnitude
    offset = (mass / (charge * field_magnitude)) * np.cross(velocity, direction)
    return position + offset


# --------------------------------------------------------------------------
# Visuals
# --------------------------------------------------------------------------
FONT = "Segoe UI"
BG = "#FFFFFF"
BORDER = "#D5D9DE"
TEXT = "#1F2933"
MUTED = "#6B7280"
ACCENT = "#1F3A5F"
BTN_BG = "#F3F4F6"
BTN_HOVER = "#E5E7EB"

COL_PATH = "#1F5FA8"
COL_GC = "#8A94A0"
COL_ARROW = "#C0392B"
COL_MARKER = "#E67E22"

PANEL_WIDTH = 310
VIEW_ZOOM = 1.15         
FIXED_SPAN = 8.0         
AXIS_TICKS = [-8, -4, 0, 4, 8]


class AuroralSimulationApp:
    def __init__(self, root):
        self.root = root
        root.title("Mapping the Magnetosphere")
        root.geometry("1320x820")
        root.minsize(1000, 700)
        root.configure(bg=BG)

        self.selected_energy = 5.0
        self.selected_pitch = 75.0
        self.current_storm_mode = False
        self.anim_running = True
        self.field_arrow = None

        self._build_interface()
        self._reset_path()
        self._animate_loop()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _reset_state_arrays(self):
        storm = self.current_storm_mode
        self.current_pos = initial_position.copy()
        self.current_vel = v_p[self.selected_energy][self.selected_pitch].copy()

        first_center = get_guiding_center(self.current_pos, self.current_vel, storm)
        self.pos_history = deque([self.current_pos.copy()], maxlen=TRAIL_SAMPLES)
        self.gc_history = deque([first_center], maxlen=TRAIL_SAMPLES)
        self.sim_time = 0.0
        self.frame_count = 0

        b0 = get_B_field(self.current_pos, storm)
        self.r0 = calculate_larmor_radius(self.current_vel, b0)
        self.span = FIXED_SPAN   

        self.r_t = [0.0]
        self.r_val = [self.r0]
        self.r_stride = 1

    # ------------------------------------------------------------------
    # Interface 
    # ------------------------------------------------------------------
    def _build_interface(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self._build_3d_view()
        self._build_control_panel()

    def _build_3d_view(self):
        self.plot_frame = tk.Frame(self.root, bg=BG)
        self.plot_frame.grid(row=0, column=0, sticky="nsew")

        self.fig = Figure(facecolor=BG)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 0.93], projection="3d")
        self._style_3d_axes()

        self.field_line_plot, = self.ax.plot([], [], [], color=COL_GC, linewidth=1.4)
        self.trajectory_line, = self.ax.plot([], [], [], color=COL_PATH, linewidth=1.8)
        self.particle_marker, = self.ax.plot(
            [0.0], [0.0], [0.0], marker="o", markersize=9, linestyle="none",
            markerfacecolor=COL_MARKER, markeredgecolor="white",
            markeredgewidth=1.2)

        self.fig.text(
            0.025, 0.965, "Mapping the Magnetosphere: A Simulation of the Aurora",
            fontsize=14, fontweight="bold", color=TEXT, family=FONT, va="center")
        self.fig.text(
            0.025, 0.935, "Axes in metres, relative to the electron's guiding centre.",
            fontsize=9, color=MUTED, family=FONT, va="center")

        handles = [
            Line2D([0], [0], color=COL_PATH, lw=2, label="Electron path"),
            Line2D([0], [0], color=COL_GC, lw=1.5, label="Guiding centre"),
            Line2D([0], [0], color=COL_ARROW, lw=2.2,
                   label="Magnetic field direction"),
            Line2D([0], [0], marker="o", color="w", markersize=9,
                   markerfacecolor=COL_MARKER, label="Electron (current)")]
        self.fig.legend(
            handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985),
            frameon=False, fontsize=9, labelcolor=TEXT)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_control_panel(self):
        tk.Frame(self.root, bg=BORDER, width=1).grid(row=0, column=1, sticky="ns")

        self.panel = tk.Frame(self.root, bg=BG, width=PANEL_WIDTH)
        self.panel.grid(row=0, column=2, sticky="ns", padx=16, pady=14)
        self.panel.pack_propagate(False)

        self._section("FIELD CONDITION", first=True)
        self.field_buttons = self._segmented(
            [(False, "Quiet"), (True, "Storm")], self._set_storm)

        self._section("INITIAL ELECTRON ENERGY")
        self.energy_buttons = self._segmented(
            [(1.0, "1 keV"), (3.0, "3 keV"), (5.0, "5 keV")], self._set_energy)

        self._section("INITIAL PITCH ANGLE")
        self.pitch_buttons = self._segmented(
            [(30.0, "30°"), (45.0, "45°"), (75.0, "75°")], self._set_pitch)

        self._section("SIMULATION")
        row = tk.Frame(self.panel, bg=BG)
        row.pack(fill="x")
        row.columnconfigure((0, 1, 2), weight=1, uniform="sim")
        sim_buttons = (("Reset", self._reset_path), ("Play", self._play_sim),
                       ("Pause", self._pause_sim))
        for col, (label, command) in enumerate(sim_buttons):
            button = self._make_button(row, label, command)
            button.grid(row=0, column=col, sticky="ew",
                        padx=(0 if col == 0 else 4, 0))

        self._build_telemetry_card()
        self._build_radius_plot()

    def _style_3d_axes(self):
        ax = self.ax
        ax.set_facecolor(BG)
        ax.view_init(elev=16, azim=-52)
        try:
            ax.set_box_aspect((1, 1, 1), zoom=VIEW_ZOOM)
        except TypeError:
            ax.set_box_aspect((1, 1, 1))

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            except Exception:
                pass
            axis.set_major_locator(FixedLocator(AXIS_TICKS))
            axis.set_major_formatter(FormatStrFormatter("%g"))
            axis.label.set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("x (m)", fontsize=9, labelpad=4)
        ax.set_ylabel("y (m)", fontsize=9, labelpad=4)
        ax.set_zlabel("z (m)", fontsize=9, labelpad=2)

    def _section(self, title, first=False):
        label = tk.Label(self.panel, text=title, font=(FONT, 8, "bold"),
                         fg=MUTED, bg=BG, anchor="w")
        label.pack(fill="x", pady=(0 if first else 12, 4))

    def _make_button(self, parent, text, command):
        return tk.Button(
            parent, text=text, command=command, font=(FONT, 9),
            relief="flat", bd=0, bg=BTN_BG, fg=TEXT,
            activebackground=BTN_HOVER, activeforeground=TEXT,
            highlightthickness=1, highlightbackground=BORDER,
            cursor="hand2", pady=6)

    def _segmented(self, options, command):
        frame = tk.Frame(self.panel, bg=BG)
        frame.pack(fill="x")
        frame.columnconfigure(tuple(range(len(options))), weight=1, uniform="seg")
        buttons = {}
        for col, (value, label) in enumerate(options):
            button = self._make_button(frame, label, lambda v=value: command(v))
            button.grid(row=0, column=col, sticky="ew",
                        padx=(0 if col == 0 else 4, 0))
            buttons[value] = button
        return buttons

    def _build_telemetry_card(self):
        self._section("TELEMETRY")
        card = tk.Frame(self.panel, bg=BG, highlightthickness=1,
                        highlightbackground=BORDER, highlightcolor=BORDER)
        card.pack(fill="x")
        card.columnconfigure(1, weight=1)

        rows = (("field", "Field condition"), ("energy", "Initial energy"),
                ("pitch", "Initial pitch angle"), ("b", "|B|"),
                ("larmor", "Larmor radius"), ("time", "Elapsed time"))
        self.tele = {}
        for i, (key, label) in enumerate(rows):
            pad_top = 10 if i == 0 else 3
            pad_bottom = 10 if i == len(rows) - 1 else 3
            name = tk.Label(card, text=label, font=(FONT, 9), fg=MUTED, bg=BG,
                            anchor="w")
            name.grid(row=i, column=0, sticky="w", padx=(12, 6),
                      pady=(pad_top, pad_bottom))
            value = tk.Label(card, text="--", font=(FONT, 9, "bold"), fg=TEXT,
                             bg=BG, anchor="e")
            value.grid(row=i, column=1, sticky="e", padx=(6, 12),
                       pady=(pad_top, pad_bottom))
            self.tele[key] = value

    def _build_radius_plot(self):
        self._section("LARMOR RADIUS OVER TIME")
        card = tk.Frame(self.panel, bg=BG, highlightthickness=1,
                        highlightbackground=BORDER, highlightcolor=BORDER)
        card.pack(fill="both", expand=True)

        self.r_fig = Figure(figsize=(3.0, 2.0), dpi=100, facecolor=BG)
        self.r_ax = self.r_fig.add_subplot(111)
        self.r_fig.subplots_adjust(left=0.27, right=0.95, top=0.95, bottom=0.24)
        ax = self.r_ax
        ax.set_facecolor(BG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=7, length=3)
        ax.grid(True, color="#ECEEF1", linewidth=0.6)
        ax.set_xlabel("time (µs)", fontsize=8, color=MUTED, labelpad=2)
        ax.set_ylabel("Larmor radius (m)", fontsize=8, color=MUTED, labelpad=2)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        self.r_line, = ax.plot([], [], color=ACCENT, linewidth=1.5)

        self.r_canvas = FigureCanvasTkAgg(self.r_fig, master=card)
        self.r_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Telemetry 
    # ------------------------------------------------------------------
    @staticmethod
    def _style_selected(button, selected):
        if selected:
            button.config(bg=ACCENT, fg="white", activebackground=ACCENT,
                          activeforeground="white", highlightbackground=ACCENT)
        else:
            button.config(bg=BTN_BG, fg=TEXT, activebackground=BTN_HOVER,
                          activeforeground=TEXT, highlightbackground=BORDER)

    def _update_button_highlights(self):
        for value, button in self.field_buttons.items():
            self._style_selected(button, value == self.current_storm_mode)
        for value, button in self.energy_buttons.items():
            self._style_selected(button, value == self.selected_energy)
        for value, button in self.pitch_buttons.items():
            self._style_selected(button, value == self.selected_pitch)

    def _update_static_telemetry(self):
        field_text = "Storm" if self.current_storm_mode else "Quiet"
        self.tele["field"].config(text=field_text)
        self.tele["energy"].config(text=f"{self.selected_energy:g} keV")
        self.tele["pitch"].config(text=f"{self.selected_pitch:g}°")

    def _update_telemetry(self, record=True):
        magnetic = get_B_field(self.current_pos, self.current_storm_mode)
        field_magnitude = np.linalg.norm(magnetic)
        radius = calculate_larmor_radius(self.current_vel, magnetic)
        self.tele["b"].config(text=f"{field_magnitude * 1e6:.4f} µT")
        self.tele["larmor"].config(text=f"{radius:.4f} m")
        self.tele["time"].config(text=f"{self.sim_time * 1e6:.2f} µs")

        if record and self.frame_count % self.r_stride == 0:
            self.r_t.append(self.sim_time * 1e6)
            self.r_val.append(radius)
            if len(self.r_t) > 2000:      # thin out, keep the whole run
                self.r_t = self.r_t[::2]
                self.r_val = self.r_val[::2]
                self.r_stride *= 2

    def _update_radius_plot(self):
        t = np.asarray(self.r_t)
        r = np.asarray(self.r_val)
        self.r_line.set_data(t, r)
        self.r_ax.set_xlim(0.0, max(t[-1], 0.05) * 1.02)
        low, high = r.min(), r.max()
        pad = max((high - low) * 0.2, 5e-5)
        self.r_ax.set_ylim(low - pad, high + pad)
        self.r_canvas.draw_idle()

    # ------------------------------------------------------------------
    # 3D view
    # ------------------------------------------------------------------
    def _apply_view(self):
        span = self.span
        self.ax.set_xlim(-span, span)
        self.ax.set_ylim(-span, span)
        self.ax.set_zlim(-span, span)
        self._update_field_arrow()

    def _update_field_arrow(self):
        if self.field_arrow is not None:
            try:
                self.field_arrow.remove()
            except Exception:
                pass
            self.field_arrow = None

        magnetic = get_B_field(self.gc_history[-1], self.current_storm_mode)
        field_magnitude = np.linalg.norm(magnetic)
        if field_magnitude == 0.0:
            return
        dx, dy, dz = magnetic / field_magnitude
        self.field_arrow = self.ax.quiver(
            0.0, 0.0, 0.0, dx, dy, dz, length=0.55 * self.span, normalize=True,
            color=COL_ARROW, linewidth=2.2, arrow_length_ratio=0.2)

    def _update_3d_view(self):
        history = np.asarray(self.pos_history)
        guiding_centers = np.asarray(self.gc_history)
        latest_center = guiding_centers[-1]
        relative_history = history - latest_center
        relative_centers = guiding_centers - latest_center

        # Only draw what is inside the box, so nothing spills past the axes.
        inside_h = np.all(np.abs(relative_history) <= self.span, axis=1)
        inside_c = np.all(np.abs(relative_centers) <= self.span, axis=1)
        path = relative_history[inside_h]
        centers = relative_centers[inside_c]

        self.trajectory_line.set_data(path[:, 0], path[:, 1])
        self.trajectory_line.set_3d_properties(path[:, 2])
        self.field_line_plot.set_data(centers[:, 0], centers[:, 1])
        self.field_line_plot.set_3d_properties(centers[:, 2])

        now = relative_history[-1]
        self.particle_marker.set_data([now[0]], [now[1]])
        self.particle_marker.set_3d_properties([now[2]])

        if self.frame_count % 25 == 0:
            self._update_field_arrow()

    # ------------------------------------------------------------------
    # Animation loop
    # ------------------------------------------------------------------
    def _animate_loop(self):
        if self.anim_running:
            storm = self.current_storm_mode
            for i in range(STEPS_PER_FRAME):
                self.current_pos, self.current_vel = euler_step(
                    self.current_pos, self.current_vel, storm)
                if i % SAMPLE_EVERY == 0:
                    self.pos_history.append(self.current_pos.copy())
                    self.gc_history.append(get_guiding_center(
                        self.current_pos, self.current_vel, storm))

            self.sim_time += STEPS_PER_FRAME * dt
            self.frame_count += 1

            self._update_3d_view()
            self._update_telemetry()
            if self.frame_count % 5 == 0:
                self._update_radius_plot()
            self.canvas.draw_idle()
        self.root.after(20, self._animate_loop)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def _reset_path(self):
        self._reset_state_arrays()
        self._update_button_highlights()
        self._update_static_telemetry()
        self._apply_view()
        self._update_3d_view()
        self._update_telemetry(record=False)
        self._update_radius_plot()
        self.canvas.draw_idle()

    def _set_storm(self, value):
        self.current_storm_mode = bool(value)
        self._reset_path()

    def _set_energy(self, value):
        self.selected_energy = value
        self._reset_path()

    def _set_pitch(self, value):
        self.selected_pitch = value
        self._reset_path()

    def _play_sim(self):
        self.anim_running = True

    def _pause_sim(self):
        self.anim_running = False


if __name__ == "__main__":
    root = tk.Tk()
    AuroralSimulationApp(root)
    root.mainloop()
