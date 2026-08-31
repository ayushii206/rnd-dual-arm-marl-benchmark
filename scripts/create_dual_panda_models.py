from pathlib import Path
import xml.etree.ElementTree as ET


SOURCE = Path(
    "assets/models/custom_panda/panda_with_site.xml"
)

OUTPUT_DIR = Path(
    "assets/models/dual_panda"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prefix_model(source_path, output_path, prefix):

    tree = ET.parse(source_path)
    root = tree.getroot()
    # Generated Panda files are stored in dual_panda/, while the
    # validated mesh assets remain in custom_panda/assets/.
    compiler = root.find("compiler")

    if compiler is not None:
        compiler.set("meshdir", "../custom_panda/assets")

    # Object types whose names should become unique
    rename_tags = {
        "body",
        "joint",
        "site",
        "camera",
        "light",
        "key",
        "sensor",

        # Tendon types
        "fixed",
        "spatial",

        # Actuator types
        "general",
        "motor",
        "position",
        "velocity",
        "intvelocity",
        "damper",
        "cylinder",
        "muscle",
        "adhesion",
    }

    # First build a mapping from old names to new names.
    name_map = {}

    for elem in root.iter():

        if "name" in elem.attrib:

            old_name = elem.attrib["name"]

            # Do not rename asset names such as meshes/materials.
            if elem.tag in rename_tags:
                new_name = f"{prefix}_{old_name}"
                name_map[old_name] = new_name
                elem.attrib["name"] = new_name

    # Update references such as joint="joint1".
    reference_attributes = [
        "joint",
        "joint1",
        "joint2",
        "body",
        "body1",
        "body2",
        "site",
        "site1",
        "site2",
        "geom",
        "geom1",
        "geom2",
        "objname",
        "tendon",
    ]

    for elem in root.iter():

        for attr in reference_attributes:

            if attr in elem.attrib:

                value = elem.attrib[attr]

                if value in name_map:
                    elem.attrib[attr] = name_map[value]

    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True
    )

    print(f"Created: {output_path}")


prefix_model(
    SOURCE,
    OUTPUT_DIR / "left_panda.xml",
    "left"
)

prefix_model(
    SOURCE,
    OUTPUT_DIR / "right_panda.xml",
    "right"
)
