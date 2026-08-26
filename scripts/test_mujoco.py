import mujoco

xml = """
<mujoco>
    <worldbody>
        <geom type="plane" size="2 2 0.1"/>
        <body pos="0 0 1">
            <joint type="free"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("MuJoCo model loaded successfully.")
print("Number of generalized coordinates:", model.nq)

for _ in range(1000):
    mujoco.mj_step(model, data)

print("Simulation completed successfully.")
print("Final object position:", data.qpos[:3])
