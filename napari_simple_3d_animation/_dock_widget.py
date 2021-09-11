"""
A napari widget for animating simple camera motions in 3D and exporting movies.
"""
import math
import time

import ffmpeg
from magicgui import magicgui
from napari_plugin_engine import napari_hook_implementation
import napari
import numpy as np


class AnimWidget:
    _worker = None

    def __init__(self) -> None:
        pass

    def cancel(self):
        napari.utils.notifications.show_info("Ending animation")
        if self._worker is not None:
            self._worker.quit()
        self._worker = None

    def camera_animation_widget(self):
        angle_widget = {"widget_type": "Slider", "min": -360, "max": 360}

        MOTION = {
            "sin": lambda p: math.sin(p * 2.0 * math.pi),
            "lerp": lambda p: p,
            "sigmoid": lambda p: expit(p * 10.0 - 10.0),
        }

        @magicgui(
            call_button="Toggle animation",
            x_deg={"value": 22.5, **angle_widget},
            y_deg=angle_widget,
            z_deg=angle_widget,
            motion={"choices": MOTION.keys()},
            target_fps={"min": 1, "max": 120},
            repetitions={"widget_type": "FloatSpinBox", "max": float("inf")},
        )
        def _widget(
            viewer: napari.Viewer,
            x_deg,
            y_deg,
            z_deg,
            motion,
            duration=10.0,
            repetitions=float("inf"),
            target_fps=60.0,
            record_movie=False,
            gui_in_movie=False,
            movie_file="animation.mp4",
        ):
            from napari.qt.threading import GeneratorWorker

            if self._worker is not None:
                self.cancel()
                return

            worker = GeneratorWorker(
                anim_worker,
                viewer.camera.angles,
                x_deg=x_deg,
                y_deg=y_deg,
                z_deg=z_deg,
                duration=duration,
                repetitions=repetitions,
                interval=1.0 / target_fps,
                f=MOTION[motion],
            )
            self._worker = worker
            worker.yielded.connect(lambda x: setattr(viewer.camera, "angles", x))
            orig_angles = viewer.camera.angles
            reset_angles = lambda: setattr(viewer.camera, "angles", orig_angles)
            worker.aborted.connect(reset_angles)
            worker.finished.connect(reset_angles)
            worker.finished.connect(self.cancel)

            if record_movie:
                f = viewer.screenshot(canvas_only=not gui_in_movie)
                width = (f.shape[1] // 2) * 2
                height = (f.shape[0] // 2) * 2
                writer = MovieWriter(width, height, target_fps, movie_file)

                def write_frame():
                    self._worker.pause()
                    frame = viewer.screenshot(canvas_only=not gui_in_movie)
                    writer.write_frame(frame)
                    self._worker.resume()

                writer.write_frame(f)
                worker.yielded.connect(lambda _: write_frame())
                worker.aborted.connect(writer.finish)
                worker.finished.connect(writer.finish)

            worker.start()

        return _widget


def anim_worker(
    angles, x_deg, y_deg, z_deg, duration, repetitions, interval=0.016, f=math.sin
):
    from napari._vispy.quaternion import quaternion2euler
    from vispy.util.quaternion import Quaternion

    p = 0.0

    orig_q = Quaternion.create_from_euler_angles(*angles, degrees=True)
    while p < repetitions:
        p += interval / duration
        x1 = f(p)
        a = [x_deg * x1, y_deg * x1, z_deg * x1]
        a = Quaternion.create_from_euler_angles(*a, degrees=True)
        a = orig_q * a
        yield (quaternion2euler(a, degrees=True))
        time.sleep(interval)


class MovieWriter:
    def __init__(self, width, height, fps, filename, vcodec="libx264") -> None:
        self.width = width
        self.height = height
        self._process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s="{}x{}".format(width, height),
                r=fps,
            )
            .output(filename, pix_fmt="yuv420p", vcodec=vcodec, r=fps)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def write_frame(self, frame):
        frame = frame[0 : self.height, 0 : self.width, 0:3]
        self._process.stdin.write(frame.astype(np.uint8).tobytes())

    def finish(self):
        self._process.stdin.close()
        self._process.wait()


def expit(x):
    return 1.0 / (1.0 + math.exp(-x))


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    widget = AnimWidget()
    return widget.camera_animation_widget
