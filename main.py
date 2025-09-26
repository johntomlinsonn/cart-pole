import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path("cartpole.xml")
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)



while True:
    mujoco.mj_step(model, data)
    viewer.render()