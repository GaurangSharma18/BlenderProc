# BlenderProc

## For Bop Data Generation
blenderproc run examples/datasets/bop_challenge/main_lm_upright.py examples/thesis/bop_dataset examples/thesis/cc_textures  examples/thesis/bop_challange/output --num_scenes=2  examples/thesis/colours

## To Visualize BOP hdf5 files
blenderproc vis hdf5 examples/thesis/bop_challange/output\bop_data\0.hdf5

## To Run custom data generation script
blenderproc run examples/thesis/project/main.py examples/thesis/resources/colours.obj examples/thesis/output examples/thesis/bop_dataset examples/thesis/colours --num_scenes=200 examples/thesis/cc_textures
