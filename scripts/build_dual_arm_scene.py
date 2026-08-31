from pathlib import Path
import copy
import xml.etree.ElementTree as ET

LEFT = Path("assets/models/dual_panda/left_panda.xml")
RIGHT = Path("assets/models/dual_panda/right_panda.xml")
OUTPUT = Path("assets/models/dual_panda/dual_arm_scene.xml")

left_tree = ET.parse(LEFT)
right_tree = ET.parse(RIGHT)

left_root = left_tree.getroot()
right_root = right_tree.getroot()

# Use left model as the base because it already contains
# compiler, defaults, assets, etc.
combined_root = copy.deepcopy(left_root)

combined_root.set("model", "dual panda cooperative transport")

# Find shared sections
combined_worldbody = combined_root.find("worldbody")
# Position the left Panda
if combined_worldbody is not None:
    for child in combined_worldbody:
        if child.attrib.get("name") == "left_link0":
            child.set("pos", "0 0.65 0")
            child.set("quat", "0.7071068 0 0 -0.7071068")
combined_actuator = combined_root.find("actuator")

combined_contact = combined_root.find("contact")

combined_tendon = combined_root.find("tendon")

combined_equality = combined_root.find("equality")


right_worldbody = right_root.find("worldbody")

right_actuator = right_root.find("actuator")

right_contact = right_root.find("contact")

right_tendon = right_root.find("tendon")

right_equality = right_root.find("equality")

# Append right robot bodies with modified base placement
if right_worldbody is not None and combined_worldbody is not None:

    for child in right_worldbody:

        new_child = copy.deepcopy(child)

        # Position the right Panda on the opposite side
        if new_child.attrib.get("name") == "right_link0":
            new_child.set("pos", "0 -0.65 0")
            new_child.set("quat", "0.7071068 0 0 0.7071068")

        combined_worldbody.append(new_child)

# Append right robot actuators

if right_actuator is not None and combined_actuator is not None:

    for child in right_actuator:

        combined_actuator.append(
            copy.deepcopy(child)
        )


# Append right robot contact exclusions

if right_contact is not None and combined_contact is not None:

    for child in right_contact:

        combined_contact.append(
            copy.deepcopy(child)
        )


# Append right robot tendons

if right_tendon is not None and combined_tendon is not None:

    for child in right_tendon:

        combined_tendon.append(
            copy.deepcopy(child)
        )


# Append right robot equality constraints

if right_equality is not None and combined_equality is not None:

    for child in right_equality:

        combined_equality.append(
            copy.deepcopy(child)
        )

# Add a floor
if combined_worldbody is not None:
    floor = ET.Element(
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "3 3 0.1",
            "rgba": "0.85 0.85 0.85 1",
        },
    )
    combined_worldbody.append(floor)

# Save
tree = ET.ElementTree(combined_root)
ET.indent(tree, space="  ")

tree.write(
    OUTPUT,
    encoding="utf-8",
    xml_declaration=True
)

print(f"Created {OUTPUT}")
