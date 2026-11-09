# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:51:21 2026

@author: abiga
"""
# -*- coding: utf-8 -*-
"""Mapping the Magnetosphere with a stable Larmor-radius calculation."""

import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

k = 1e-7
q = 1.602176634e-19
m_e = 9.1093837015e-31
m_p = 1.67262192369e-27
R_E = 6.371e6
M_dipole = 8.0e22
tilt = np.radians(23.4)
dt = 1.5e-11

M_dipole_vec = np.array(
    [-M_dipole * np.sin(tilt), 0.0, -M_dipole * np.cos(tilt)]
)
start_altitude = 0.0
initial_position = np.array(
    [
        start_altitude * np.sin(tilt),
        0.0,
        R_E + start_altitude * np.cos(tilt),
    ]
)


def calculate_velocity_matrix(active_mass, is_proton=False):
    velocity_matrix = {}
    for energy_kev in (1.0, 3.0, 5.0):
        speed = np.sqrt(2.0 * energy_kev * 1000.0 * q / active_mass)
        velocity_matrix[energy_kev] = {}

        for pitch_deg in (30.0, 45.0, 75.0):
            alpha = np.radians(pitch_deg)
            radius = np.linalg.norm(initial_position)
            dot_product = np.dot(M_dipole_vec, initial_position)
            magnetic = k * (
                3.0 * initial_position * dot_product
                - M_dipole_vec * radius**2
            ) / radius**5
            direction = magnetic / np.linalg.norm(magnetic)

            points_outward = np.dot(direction, initial_position) > 0.0
            if points_outward != is_proton:
                direction = -direction

            reference = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(direction, reference)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            perpendicular = np.cross(direction, reference)
            perpendicular /= np.linalg.norm(perpendicular)

            velocity_matrix[energy_kev][pitch_deg] = (
                speed * np.cos(alpha) * direction
                + speed * np.sin(alpha) * perpendicular
            )
    return velocity_matrix


v_p = calculate_velocity_matrix(m_e)


def get_B_field(position, storm_mode=False):
    position = np.asarray(position, dtype=float)
    radius = np.linalg.norm(position)
    if radius == 0.0:
        return np.zeros(3)

    dot_product = np.dot(M_dipole_vec, position)
    magnetic = k * (
        3.0 * position * dot_product - M_dipole_vec * radius**2
    ) / radius**5
    if storm_mode:
        magnetic[2] -= 0.6e-6
    return magnetic


def calculate_larmor_radius(velocity, magnetic, mass=m_e):
    field_magnitude = np.linalg.norm(magnetic)
    if field_magnitude == 0.0:
        return 0.0
    field_direction = magnetic / field_magnitude
    perpendicular_velocity = (
        velocity - np.dot(velocity, field_direction) * field_direction
    )
    return (
        mass * np.linalg.norm(perpendicular_velocity)
        / (q * field_magnitude)
    )


def calculate_physics_step(position, velocity, storm_mode=False, is_proton=False):
    magnetic = get_B_field(position, storm_mode)
    field_magnitude = np.linalg.norm(magnetic)
    mass = m_p if is_proton else m_e
    charge = q if is_proton else -q
    acceleration = (charge / mass) * np.cross(velocity, magnetic)

    current = q**2 * field_magnitude / (2.0 * np.pi * mass)
    larmor_radius = calculate_larmor_radius(velocity, magnetic, mass)
    induced_field = (
        (4.0 * np.pi * k) * current / (2.0 * larmor_radius)
        if larmor_radius > 0.0
        else 0.0
    )
    return acceleration, current, induced_field, larmor_radius


def boris_rotate(velocity, magnetic, charge, mass, time_step):
    """Rotate velocity without adding magnetic-field numerical energy."""
    t = (charge / mass) * magnetic * (time_step / 2.0)
    s = 2.0 * t / (1.0 + np.dot(t, t))
    velocity_prime = velocity + np.cross(velocity, t)
    return velocity + np.cross(velocity_prime, s)


def get_guiding_center(position, velocity, storm_mode=False, is_proton=False):
    magnetic = get_B_field(position, storm_mode)
    field_magnitude = np.linalg.norm(magnetic)
    if field_magnitude == 0.0:
        return position.copy()
    mass = m_p if is_proton else m_e
    charge = q if is_proton else -q
    direction = magnetic / field_magnitude
    return position + (mass / (charge * field_magnitude)) * np.cross(
        velocity, direction
    )


class AuroralSimulationApp:
    def __init__(self, root):
        self.root = root
        root.title("Mapping the Magnetosphere")
        root.geometry("1180x800")
        root.configure(bg="#E3F2FD")

        self.selected_energy = 5.0
        self.selected_pitch = 75.0
        self.current_storm_mode = False
        self.anim_running = True
        self.c_selected = "#e3c4ee"
        self.c_default = "#e6d8f1"

        self._build_interface()
        self._reset_state_arrays()
        self._update_button_highlights()
        self._animate_loop()

    def _reset_state_arrays(self):
        self.current_pos = initial_position.copy()
        self.current_vel = v_p[self.selected_energy][self.selected_pitch].copy()
        self.pos_history = [self.current_pos.copy()]
        self.gc_history = [
            get_guiding_center(
                self.current_pos, self.current_vel, self.current_storm_mode
            )
        ]

    def _build_interface(self):
        self.left_frame = tk.Frame(self.root, bg="#E3F2FD")
        self.left_frame.place(x=20, y=20, width=780, height=740)
        self.fig = Figure(figsize=(7.5, 7.3), facecolor="#E3F2FD")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#FFFFFF")
        self.ax.set_axis_off()
        self.field_line_plot, = self.ax.plot(
            [], [], [], color="#566573", alpha=0.9, linewidth=1.2,
            label="Guiding Center",
        )
        self.trajectory_line, = self.ax.plot(
            [], [], [], color="#9B59B6", linewidth=2.0, label="Particle Path"
        )
        self.span = 15.0
        self.ax.view_init(elev=10, azim=45)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.get_tk_widget().place(x=0, y=0, width=780, height=735)

        self.right_frame = tk.Frame(self.root, bg="#E3F2FD")
        self.right_frame.place(x=820, y=20, width=320, height=740)
        tk.Label(
            self.right_frame, text="Magnetic Field Strength:",
            font=("Georgia", 11, "bold"), bg="#E3F2FD",
        ).place(x=10, y=10)
        self.btn_quiet = tk.Button(
            self.right_frame, text="Quiet", command=self._set_quiet
        )
        self.btn_quiet.place(x=10, y=45, width=110, height=40)
        self.btn_storm = tk.Button(
            self.right_frame, text="Storm", command=self._set_storm
        )
        self.btn_storm.place(x=140, y=45, width=110, height=40)

        tk.Label(
            self.right_frame, text="Energy:",
            font=("Georgia", 11, "bold"), bg="#E3F2FD",
        ).place(x=10, y=120)
        self.energy_buttons = {}
        for energy, x in ((1.0, 10), (3.0, 100), (5.0, 190)):
            button = tk.Button(
                self.right_frame, text=f"{energy:g} keV",
                command=lambda value=energy: self._set_energy(value),
            )
            button.place(x=x, y=155, width=80, height=40)
            self.energy_buttons[energy] = button

        tk.Label(
            self.right_frame, text="Pitch Angle:",
            font=("Georgia", 11, "bold"), bg="#E3F2FD",
        ).place(x=10, y=240)
        self.pitch_buttons = {}
        for pitch, x in ((30.0, 10), (45.0, 100), (75.0, 190)):
            button = tk.Button(
                self.right_frame, text=f"{pitch:g} deg",
                command=lambda value=pitch: self._set_pitch(value),
            )
            button.place(x=x, y=275, width=80, height=40)
            self.pitch_buttons[pitch] = button

        tk.Button(
            self.right_frame, text="Reset", command=self._reset_path
        ).place(x=10, y=380, width=80, height=40)
        tk.Button(
            self.right_frame, text="Play", command=self._play_sim
        ).place(x=100, y=380, width=80, height=40)
        tk.Button(
            self.right_frame, text="Pause", command=self._pause_sim
        ).place(x=190, y=380, width=80, height=40)
        self.lbl_telemetry = tk.Label(
            self.right_frame, text="|B|: ----\nLarmor radius: ----",
            font=("Georgia", 10, "bold"), bg="#e6d8f1", justify="left",
        )
        self.lbl_telemetry.place(x=10, y=450)

    def _update_button_highlights(self):
        self.btn_quiet.config(
            bg=self.c_selected if not self.current_storm_mode else self.c_default
        )
        self.btn_storm.config(
            bg=self.c_selected if self.current_storm_mode else self.c_default
        )
        for energy, button in self.energy_buttons.items():
            button.config(
                bg=self.c_selected
                if self.selected_energy == energy
                else self.c_default
            )
        for pitch, button in self.pitch_buttons.items():
            button.config(
                bg=self.c_selected
                if self.selected_pitch == pitch
                else self.c_default
            )

    def _animate_loop(self):
        if self.anim_running:
            charge = -q
            for _ in range(320):
                magnetic = get_B_field(
                    self.current_pos, self.current_storm_mode
                )
                old_velocity = self.current_vel
                self.current_vel = boris_rotate(
                    old_velocity, magnetic, charge, m_e, dt
                )
                self.current_pos += (
                    0.5 * (old_velocity + self.current_vel) * dt
                )
                self.pos_history.append(self.current_pos.copy())
                self.gc_history.append(
                    get_guiding_center(
                        self.current_pos,
                        self.current_vel,
                        self.current_storm_mode,
                    )
                )

            history = np.asarray(self.pos_history)
            guiding_centers = np.asarray(self.gc_history)
            latest_center = guiding_centers[-1]
            relative_history = history - latest_center
            relative_centers = guiding_centers - latest_center
            self.trajectory_line.set_data(
                relative_history[:, 0], relative_history[:, 1]
            )
            self.trajectory_line.set_3d_properties(relative_history[:, 2])
            self.field_line_plot.set_data(
                relative_centers[:, 0], relative_centers[:, 1]
            )
            self.field_line_plot.set_3d_properties(relative_centers[:, 2])

            _, _, _, radius = calculate_physics_step(
                self.current_pos, self.current_vel, self.current_storm_mode
            )
            field_magnitude = np.linalg.norm(
                get_B_field(self.current_pos, self.current_storm_mode)
            )
            self.lbl_telemetry.config(
                text=f"|B|: {field_magnitude * 1e6:.4f} µT\n"
                f"Larmor radius: {radius:.2f} m"
            )
            self.ax.set_xlim([-self.span, self.span])
            self.ax.set_ylim([-self.span, self.span])
            self.ax.set_zlim([-self.span, self.span])
            self.canvas.draw_idle()
        self.root.after(20, self._animate_loop)

    def _reset_path(self):
        self._reset_state_arrays()
        self.trajectory_line.set_data([], [])
        self.trajectory_line.set_3d_properties([])
        self.field_line_plot.set_data([], [])
        self.field_line_plot.set_3d_properties([])
        self.canvas.draw_idle()

    def _set_quiet(self):
        self.current_storm_mode = False
        self._update_button_highlights()
        self._reset_path()

    def _set_storm(self):
        self.current_storm_mode = True
        self._update_button_highlights()
        self._reset_path()

    def _set_energy(self, value):
        self.selected_energy = value
        self._update_button_highlights()
        self._reset_path()

    def _set_pitch(self, value):
        self.selected_pitch = value
        self._update_button_highlights()
        self._reset_path()

    def _play_sim(self):
        self.anim_running = True

    def _pause_sim(self):
        self.anim_running = False


if __name__ == "__main__":
    root = tk.Tk()
    AuroralSimulationApp(root)
    root.mainloop()



