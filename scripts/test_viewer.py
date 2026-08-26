import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
    <worldbody>
        <geom type="plane" size="2 2 0.1"/>
        <body pos="0 0 1">
            <joint type="free"/>
            <geom type="box"
                  size="0.1 0.1 0.1"
                  rgba="0.8 0.3 0.3 1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
