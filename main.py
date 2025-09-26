import mujoco
import mujoco_viewer

model = mujoco.mj_load_xml('cartpole.xml')
data = mujoco.mj_makeData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

viewer.render()