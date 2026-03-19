import trimesh
import os

def decompose_and_generate_xml(input_filename, output_folder, scale="4.5"):
    # Load the original concave funnel mesh
    print(f"Loading {input_filename}...")
    mesh = trimesh.load(input_filename)

    # Perform convex decomposition using V-HACD
    print("Decomposing mesh... This might take a moment depending on mesh complexity.")
    # You can tweak maxhulls if the collision geometry isn't tight enough
    convex_pieces = mesh.convex_decomposition(maxConvexHulls=16)

    # Ensure the output is iterable (in case it only finds one piece)
    if not isinstance(convex_pieces, list):
        convex_pieces = [convex_pieces]

    print(f"Successfully decomposed into {len(convex_pieces)} convex parts.")

    # Create a directory to store the new hull pieces
    os.makedirs(output_folder, exist_ok=True)

    xml_output = ""

    # Save each piece and generate the Stonefish XML snippet
    for i, piece in enumerate(convex_pieces):
        part_filename = f"funnel_hull_{i}.obj"
        output_path = os.path.join(output_folder, part_filename)
        
        # Export the convex piece
        piece.export(output_path)
        
        # Build the XML string using your existing material and look definitions
        xml_output += f"""
            <external_part name="FunnelMesh_Hull_{i}" type="model" physics="submerged" buoyant="false">
                <physical>
                    <mesh filename="{output_folder}/{part_filename}" scale="{scale}"/>
                    <origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0"/>
                </physical>
                <material name="Aluminium"/>
                <look name="gray"/>
                <compound_transform rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0"/>
            </external_part>"""

    print("\n" + "="*50)
    print("SUCCESS! Replace your single FunnelMesh <external_part> in the XML with the following:")
    print("="*50)
    print(xml_output)
    print("="*50)

if __name__ == "__main__":
    # Ensure this matches your exact file name
    INPUT_OBJ = "funnel_new.obj" 
    OUTPUT_DIR = "funnel_parts"
    
    decompose_and_generate_xml(INPUT_OBJ, OUTPUT_DIR)