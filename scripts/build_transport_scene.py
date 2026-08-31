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
        "mass": "1.90",
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

ET.indent(tree, space="  ")

tree.write(
    OUTPUT,
    encoding="utf-8",
    xml_declaration=True,
)

print(f"Created {OUTPUT}")
