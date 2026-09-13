"""A small always-on-top Tk window with a live simulation-speed slider.

Deliberately single-threaded: Tk/Tcl objects aren't safe to drive from more
than one thread, so this window is created on and pumped from the same
thread that runs run_world.py's viewer loop (call .pump() once per loop
iteration, alongside viewer.sync()). It only changes how fast *simulated*
time is allowed to advance relative to wall-clock time; it has no connection
to BalanceBase or Arms, so the robot's commanded velocity, torque limits, and
control gains (all defined in simulated seconds) are unaffected by where the
slider sits.
"""


class SpeedSlider:
    def __init__(self, minimum=0.1, maximum=20.0, initial=1.0, title="Simulation speed"):
        import tkinter as tk

        self.minimum, self.maximum = minimum, maximum
        self._value = initial
        self._uncapped = False
        self._closed = False

        self._root = tk.Tk()
        self._root.title(title)
        self._root.attributes("-topmost", True)
        self._root.geometry("300x130")
        self._root.resizable(False, False)

        self._label = tk.Label(self._root, text=f"{initial:.1f}x wall-clock speed", font=("Segoe UI", 10))
        self._label.pack(pady=(12, 2))

        def on_slide(raw):
            self._value = float(raw)
            if not uncapped_var.get():
                self._label.config(text=f"{self._value:.1f}x wall-clock speed")

        self._scale = tk.Scale(self._root, from_=minimum, to=maximum, resolution=0.1,
                                orient="horizontal", length=260, showvalue=False, command=on_slide)
        self._scale.set(initial)
        self._scale.pack(pady=4)

        uncapped_var = tk.BooleanVar(value=False)

        def on_toggle():
            checked = uncapped_var.get()
            self._uncapped = checked
            self._scale.config(state="disabled" if checked else "normal")
            self._label.config(text="MAX (uncapped)" if checked
                                else f"{self._scale.get():.1f}x wall-clock speed")

        tk.Checkbutton(self._root, text="Uncapped (run as fast as possible)",
                       variable=uncapped_var, command=on_toggle).pack(pady=6)

        def on_close():
            self._closed = True
            self._root.destroy()

        self._root.protocol("WM_DELETE_WINDOW", on_close)

    def pump(self):
        """Process pending GUI events. Call once per simulation-loop iteration,
        from the same thread that created this SpeedSlider."""
        if self._closed:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except Exception:
            self._closed = True

    def close(self):
        if not self._closed:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._closed = True

    @property
    def closed(self):
        return self._closed

    @property
    def multiplier(self):
        """Wall-clock speed multiplier, or None when "uncapped" is checked
        (run as fast as the machine can simulate/render, no sleeping at all)."""
        return None if self._uncapped else self._value


def main():
    """Standalone smoke test: shows the slider and prints its value."""
    import time
    slider = SpeedSlider()
    print("Move the slider or check 'Uncapped'; close the window to stop.")
    try:
        while not slider.closed:
            slider.pump()
            print(f"\rmultiplier = {slider.multiplier}   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    print("\nclosed.")


if __name__ == "__main__":
    main()
