import blenderproc as bproc
import argparse
import numpy as np 
import random 
import os
from pathlib import Path
import bpy
import matplotlib
import sys
import h5py
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('scene', help="Path to the scene.obj file, should be examples/resources/scene.obj")
parser.add_argument('output_dir', help="Path to where the final files, will be saved, could be examples/basics/basic/output")
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('image_dir', help="Images to give colour to objects")
parser.add_argument('--num_scenes', type=int, default=10, help="How many scenes with 25 images each to generate")
parser.add_argument('cc_textures_path', default="resources/cctextures", help="Path to downloaded cc textures")
args = parser.parse_args()

bproc.init()


# load the objects into the scene
objs = bproc.loader.load_obj(args.scene)

# Set some category ids for loaded objects
for j, obj in enumerate(objs):
    obj.set_cp("category_id", j + 1)

'''
room_planes = [bproc.object.create_primitive('PLANE', scale=[500, 500, 500]),
               bproc.object.create_primitive('PLANE', scale=[500, 500, 500], location=[0, -500, -100], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[500, 500, 500], location=[0, 500, -100], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[500, 500, 500], location=[500, 0, -100], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[500, 500, 500], location=[-500, 0, -100], rotation=[0, 1.570796, 0])]
'''

room_planes = [bproc.object.create_primitive('PLANE', scale=[500, 500, 500])]

# sample light color and strenght from ceiling
#light_plane = bproc.object.create_primitive('PLANE', scale=[500, 500, 500], location=[1000, 1000, 500])
#light_plane.set_name('light_plane')
#light_plane_material = bproc.material.create('light_material')

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)



# Create a SUN light and set its properties
#light = bproc.types.Light()
#light.set_type("SUN")
#light.set_location([0, 0, -200])
#light.set_rotation_euler([-0.063, 0.6177, -0.1985])
#light.set_energy(1)
#light.set_color([1, 0.978, 0.407])

# Create a SUN light and set its properties
#light = bproc.types.Light()
#light.set_type("SUN")
#light.set_location([500, 500, 500])
#light.set_rotation_euler([-0.063, 0.6177, -0.1985])
#light.set_energy(500)
#light.set_color([1, 0.978, 0.407])

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_type("SUN")
light_point.set_location([500, 500, 800])
light_point.set_rotation_euler([-0.063, 0.6177, -0.1985])
light_point.set_energy(5)
light_point.set_color([1, 0.978, 0.407])

# define the camera resolution
bproc.camera.set_resolution(640, 480)
# BVH tree used for camera obstacle checks
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)
# Define a function that samples 6-DoF poses

def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-200, -200, -200], [100, 100, 100])
    max = np.random.uniform([600, 600, 600], [300, 300, 300])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# activate normal and depth rendering
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
#bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name", "cp_bop_dataset_name"], default_values={"category_id": 0, "cp_bop_dataset_name": None})

for i in range(args.num_scenes):
    
    cam_poses = 0

    while cam_poses < 30:
        # Sample location
        location = bproc.sampler.shell(center = [50, 50, 50],
                                radius_min = 200,
                                radius_max = 500, #500
                                elevation_min = 30,
                                elevation_max = 130)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(objs, size=200, replace=True))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 20}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # Entity

    images = list(Path(args.image_dir).absolute().rglob("Colours*.png"))

    # Collect all materials
    materials = bproc.material.collect_all()

    # Go through all objects
    for obj in objs:
        # For each material of the object
        #for i in range(len(obj.get_materials())):
        mat = obj.get_materials()[0]
        # In 50% of all cases
        #if np.random.uniform(0, 1) <= 0.5:
            # Replace the material with a random one
        #obj.set_material(i, random.choice(materials))
        #obj.set_location(np.random.uniform([10, 10, 5], [200, 200, 100]))
        #obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi * 2,np.pi * 2, np.pi * 2]))
        image = bpy.data.images.load(filepath=str(random.choice(images)))
        mat.set_principled_shader_value("Base Color", image)
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))

    # Sample two light sources
    #light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    #light_plane.replace_materials(light_plane_material)
    #light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    #location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5, elevation_min = 5, elevation_max = 89)
    #light_point.set_location(location)

    bproc.object.sample_poses(objects_to_sample = objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # Define a function that samples the initial pose of a given object above the ground
    def sample_initial_pose(obj: bproc.types.MeshObject):
        obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                    min_height=10, max_height=400, face_sample_range=[0.4, 0.6]))
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))

    # Sample objects on the given surface
    placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=objs,
                                                          surface=room_planes[0],
                                                          sample_pose_func=sample_initial_pose,
                                                          min_distance=10,
                                                          max_distance=200)
    # render the whole pipeline  
    data = bproc.renderer.render()

    # write the data to a .hdf5 container
    bproc.writer.write_hdf5('examples/thesis/output/seg/', data,append_to_existing_output=True)

    # Write data to coco file
    bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        color_file_format="JPEG",
                                        mask_encoding_format='polygon')


    # Write data to bop format
    bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                            target_objects = objs,
                            dataset = 'lm',
                            depth_scale = 10,
                            depths = data["depth"],
                            colors = data["colors"],
                            color_file_format = "JPEG",
                            ignore_dist_thres = 10000,
                            save_world2cam=True,
                            m2mm = True,
                            append_to_existing_output=True,
                            frames_per_chunk=1000)
