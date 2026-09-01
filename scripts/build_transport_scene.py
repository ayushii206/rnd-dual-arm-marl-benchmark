from pathlib import Path
import xml.etree.ElementTree as ET

SOURCE = Path(
    "assets/models/dual_panda/dual_arm_scene.xml"
)

OUTPUT = Path(
    "assets/models/dual_panda/transport_scene.xml"
)

tree = ET.parse(SOURCE)
root = tree.getroot()

worldbody = root.find("worldbody")

if worldbody is None:
    raise RuntimeError("worldbody not found")

# ------------------------------------------------------------------
# Gripper actuator configuration
# ------------------------------------------------------------------

def strengthen_gripper_actuators(root):
    """
    Increase the stiffness and damping of both Panda gripper
    position servos.

    The original Menagerie gripper actuator uses:
        biasprm="0 -100 -10"

    At the blocked grasp position of approximately 0.025 m,
    that produces only about 2.5 N of actuator force.

    For the cooperative transport task we increase the
    stiffness and damping by 10x:
        biasprm="0 -1000 -100"

    The existing actuator force limit is left unchanged.
    """

    actuator_section = root.find("actuator")

    if actuator_section is None:
        raise RuntimeError(
            "actuator section not found"
        )

    gripper_names = {
        "left_actuator8",
        "right_actuator8",
    }

    found_grippers = set()

    for actuator in actuator_section:

        actuator_name = actuator.get("name")

        if actuator_name not in gripper_names:
            continue

        # Scale the complete position-servo dynamics by 10x.
        #
        # Original Menagerie gripper:
        #   gainprm="0.01568627451 0 0"
        #   biasprm="0 -100 -10"
        #
        # The control gain MUST be scaled together with the
        # stiffness. Otherwise ctrl=255 would correspond to
        # only ~0.004 m instead of the intended 0.040 m.
        #
        # New mapping remains:
        #   ctrl =   0  -> target opening 0.000 m
        #   ctrl = 255  -> target opening 0.040 m
        #
        # while blocked-contact forces become approximately
        # 10x stronger.

        actuator.set(
            "gainprm",
            "0.3137254902 0 0",
        )

        actuator.set(
            "biasprm",
            "0 -2000 -200",
        )

        found_grippers.add(
            actuator_name
        )

    missing_grippers = (
        gripper_names
        - found_grippers
    )

    if missing_grippers:
        raise RuntimeError(
            "Could not find gripper actuators: "
            f"{sorted(missing_grippers)}"
        )
        
# ------------------------------------------------------------------
# Work surface
# ------------------------------------------------------------------

ET.SubElement(
    worldbody,
    "geom",
    {
        "name": "work_table",
        "type": "box",
        "pos": "0.20 0 0.45",
        "size": "0.90 0.42 0.05",
        "rgba": "0.45 0.45 0.45 1",
        "friction": "1.0 0.005 0.0001",
    },
)

# ------------------------------------------------------------------
# Transport object
# ------------------------------------------------------------------

box = ET.SubElement(
    worldbody,
    "body",
    {
        "name": "transport_object",
        "pos": "0 0 0.56",
    },
)

ET.SubElement(
    box,
    "freejoint",
    {
        "name": "object_freejoint",
    },
)

ET.SubElement(
    box,
    "geom",
    {
        "name": "transport_box_geom",
        "type": "box",
        "size": "0.20 0.12 0.06",
        "mass": "1.88",
        "friction": "1.0 0.005 0.0001",
        "rgba": "0.2 0.5 0.8 1",
    },
)

# ------------------------------------------------------------
# Grasp handles
# ------------------------------------------------------------

ET.SubElement(
    box,
    "geom",
    {
        "name": "left_grasp_handle",
        "type": "box",
        "pos": "0 0.135 0",
        "size": "0.025 0.015 0.020",
        "mass": "0.05",
        "friction": "1.5 0.005 0.0001",
        "rgba": "0.95 0.55 0.15 1",
    },
)

# Upper retaining shoulder for the left grasp handle.
#
# The main handle is 50 mm wide in X.
# This shoulder is 70 mm wide in X, creating a
# 10 mm overhang on each side.
#
# Its underside is at object-local Z = +0.010 m.
# The initial fingertip upper edge is around +0.0056 m,
# leaving roughly 4.4 mm of vertical clearance before
# the fingertip mechanically engages the shoulder.
ET.SubElement(
    box,
    "geom",
    {
        "name": "left_grasp_upper_shoulder",
        "type": "box",
        "pos": "0 0.135 0.0125",
        "size": "0.035 0.015 0.0025",
        "mass": "0.01",
        "friction": "1.5 0.005 0.0001",
        "rgba": "0.95 0.35 0.10 1",
    },
)

ET.SubElement(
    box,
    "geom",
    {
        "name": "right_grasp_handle",
        "type": "box",
        "pos": "0 -0.135 0",
        "size": "0.025 0.015 0.020",
        "mass": "0.05",
        "friction": "1.5 0.005 0.0001",
        "rgba": "0.95 0.55 0.15 1",
    },
)

# Upper retaining shoulder for the right grasp handle.
ET.SubElement(
    box,
    "geom",
    {
        "name": "right_grasp_upper_shoulder",
        "type": "box",
        "pos": "0 -0.135 0.0125",
        "size": "0.035 0.015 0.0025",
        "mass": "0.01",
        "friction": "1.5 0.005 0.0001",
        "rgba": "0.95 0.35 0.10 1",
    },
)

# Object center
ET.SubElement(
    box,
    "site",
    {
        "name": "object_center",
        "pos": "0 0 0",
        "size": "0.012",
        "rgba": "1 1 0 1",
    },
)

# Left grasp target on object
ET.SubElement(
    box,
    "site",
    {
        "name": "left_grasp_site",
        "pos": "0 0.135 0",
        "size": "0.012",
        "rgba": "1 0 0 1",
    },
)

# Right grasp target on object
ET.SubElement(
    box,
    "site",
    {
        "name": "right_grasp_site",
        "pos": "0 -0.135 0",
        "size": "0.012",
        "rgba": "0 1 0 1",
    },
)

# ------------------------------------------------------------------
# Goal region
# ------------------------------------------------------------------

ET.SubElement(
    worldbody,
    "geom",
    {
        "name": "goal_region",
        "type": "box",
        "pos": "0.45 0 0.505",
        "size": "0.22 0.15 0.005",
        "rgba": "0.2 1.0 0.2 0.30",
        "contype": "0",
        "conaffinity": "0",
    },
)

ET.SubElement(
    worldbody,
    "site",
    {
        "name": "goal_center",
        "pos": "0.45 0 0.56",
        "size": "0.015",
        "rgba": "0 1 1 1",
    },
)

# ------------------------------------------------------------------
# Strengthen gripper actuators for cooperative grasping
# ------------------------------------------------------------------

strengthen_gripper_actuators(
    root
)

ET.indent(tree, space="  ")

tree.write(
    OUTPUT,
    encoding="utf-8",
    xml_declaration=True,
)

print(f"Created {OUTPUT}")
