import time
import math
import mujoco
import mujoco.viewer

MODEL_PATH = 'humming_bird.xml'

# Simple flapping demo to validate joints and motors
if __name__ == '__main__':
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)

    # Control targets: periodic flapping and pitching
    t0 = time.time()
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            t = time.time() - t0
            # Flap frequency ~ 8 Hz (slowed way down for clarity)
            flap = 0.6 * math.sin(2 * math.pi * 8 * t)
            pitch = 0.3 * math.sin(2 * math.pi * 16 * t + math.pi/2)

            # Right/Left flap motors are first two; pitch motors next two
            # Order is determined by actuator listing in the XML
            d.ctrl[0] = flap
            d.ctrl[1] = -flap  # counter-phase for symmetric stroke
            d.ctrl[2] = pitch
            d.ctrl[3] = -pitch
        

            mujoco.mj_step(m, d)
            viewer.sync()
