import blenderproc as bproc
import argparse
import numpy as np 
import random 
import os

parser = argparse.ArgumentParser()
parser.add_argument('scene', help="Path to the scene.obj file, should be examples/resources/scene.obj")
parser.add_argument('output_dir', help="Path to where the final files, will be saved, could be examples/basics/basic/output")
args = parser.parse_args()

bproc.init()


# load the objects into the scene
objs = bproc.loader.load_obj(args.scene)

# Set some category ids for loaded objects
for j, obj in enumerate(objs):
    obj.set_cp("category_id", j + 1)


# Create a SUN light and set its properties
light = bproc.types.Light()
light.set_type("SUN")
light.set_location([0, 0, -200])
light.set_rotation_euler([-0.063, 0.6177, -0.1985])
light.set_energy(1)
light.set_color([1, 0.978, 0.407])

# Create a SUN light and set its properties
light = bproc.types.Light()
light.set_type("SUN")
light.set_location([0, 0, 500])
light.set_rotation_euler([-0.063, 0.6177, -0.1985])
light.set_energy(1)
light.set_color([1, 0.978, 0.407])


# define the camera resolution
bproc.camera.set_resolution(512, 512)

# Find point of interest, all cam poses should look towards it
poi = bproc.object.compute_poi(objs)
# Sample five camera poses
for i in range(5):
    # Sample random camera location above objects
    location = np.random.uniform([-200, -200, 400], [200, 200, 400])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


# Entity

# Collect all materials
materials = bproc.material.collect_all()

# Go through all objects
for obj in objs:
    # For each material of the object
    for i in range(len(obj.get_materials())):
        # In 50% of all cases
        #if np.random.uniform(0, 1) <= 0.5:
            # Replace the material with a random one
        obj.set_material(i, random.choice(materials))
        obj.set_location(np.random.uniform([-150, -150, 0], [150, 150, 80]))
        obj.set_rotation_euler(np.random.uniform([2, 2, 2], [-2, -2, -2]))


# add a spotlight which intersect with the sampled camera poses
spot_light = bproc.lighting.add_intersecting_spot_lights_to_camera_poses(clip_start=7, clip_end=15)

# activate normal and depth rendering
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])


# render the whole pipeline
data = bproc.renderer.render()

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)

# Write data to coco file
bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG")