I had a really boring christmas so I decided to convert my last hackathon project, GPU voxelizer, to CUDA. Inspired by Nvidia's sparse voxel octree publication by Samuli Laine and Tero Karras I took the project even further and wrote a prototype implementation of full mesh simplification computation that moves most of the work to GPU. I also wanted to see my new super expensive GPU doing some work for me :).

I did this project mostly for a personal challenge, but there might be some useful stuff and ideas for future development/prototyping. So a wall of text with some images incoming!

I ended up trying a super fascinating and simple approach that is as modular and fast as possible. Core idea of my prototype is to create a super accurate voxel octree presentation of original geometry and use it for ray casting. One voxel consist of a bit interleaved position value (morton coded) and a RGB24 color. Most optimal case would be to have voxels smaller than texels.

My naive and simple voxelizer runs mostly on GPU and it takes a list of textured triangles and returns a list of voxels. Sparse octree is then built on CPU from these voxels. This is how a viking house from unity assets looks like when raytraced with a voxel size of 3mm.

(SHOW IMAGE: viking_d12.png)
(SHOW IMAGE: viking_close_d13.png)

It takes ~5.8 seconds from my ridiculously op GPU to voxelize this building. Voxel count is ~100 000 000. Construction of the octree on CPU takes a bit longer, ~8.4 seconds. Memory usage is quite high (~1.0GB) but I didn't spend any time optimizing that.

After this preprocessing phase my unified approach really starts to shine! Finding solid voxels at the target level is basically just:

voxels = voxels.select(voxel => voxel.morton >> levelDiff * 3).unique();

where voxels-variable contains original voxels and levelDiff is difference between depths of voxel octree and target level (target level = tree depth). Face geometry generation is even simpler and can be done for each solid voxel by checking morton coordinates of 6 adjacent voxels. All data resides in flat arrays which makes it super easy and fast for gpu to generate vertex and index lists (< 1ms). Vertices (aka sample points) are moved to their final positions by shooting 256 rays per vertex against the voxel octree and finding the shortest distance. It takes ~3 seconds to generate 125k vertex and 250k triangle model.

Meshes are simplified using quadric error metrics. I wrote a super simple implementation few years back. Generated meshes are simplified quite aggressively to reduce triangle count and uv mapper's workload. It takes ~10 seconds to simplify and ~2 to generate uv map for 250k model. My mesh simplifier has few known bottlenecks which makes it quite slow.

Final phase of the computation is material sampling which happens to be my favorite :). Sampling is done against the voxel octree and for 2k x 2k texture it takes ~0.5 seconds to sample once per texel. My sampling code is super simple.

Here you can see the viking house optimized using low settings:

(SHOW IMAGE: show 3 images of house_8_5)

And here is the same building optimized using much higher settings:

(SHOW IMAGE: show 3 images of house_12_7)

Red patches means that the sampling failed. It also happens to be a good indicator of differences between original and optimized meshes. It took ~1.7 seconds to optimize the former building and ~33.8 seconds for the latter one. It takes ~60 seconds to compute 6 levels of detail using higher settings. I spent barely any time optimizing algorithms (no CPU-GPU work interleaving for example) and my code is CPU bound.

Conclusion:

It is a super fascinating idea to have a simple algorithm(s) and method(s) that unify all the different steps in our computation pipeline into a single coherent solution that runs well on gpu too. My implementation might not be usable as it is, but I really love this approach for it's simplicity and elegancy! :D

Pros of my approach/implementation):
 - I love morton coded voxels!
 - Sparse tree: more efficient memory usage
 - Super simple algorithms thanks to the unification
 - Preprocessing is streamable and it removes the need of having original data in store
 - Voxel octree works as an intermediate format too
 - Separation of voxels and data: voxels can have any kind of payload with them
 - No dependencies between differents steps so the process should be splittable to different service modules
 - Easy to fit to gpu

Cons:
 - Memory vs performance
 - CPU is still a bottleneck in some cases
 - My ray-octree intersection code could be faster