Accel
===================

This repository contains the functionality for two types of acceleration structures.  A uniform grid for accelerating nearest neighbor searches, and a KD Tree for accelerating ray tracking.  Both are run on the GPU using CUDA.  This project is a joint effort between Danny Rerucha and I for the GPU Programming Course at the University of Pennsylvania (CIS565).

Nearest Neighbor Search and the Uniform Grid
----------
Before we get to our discussion of the uniform grid, let's talk about nearest neighbor search for a second.  It's hopefully somewhat clear what we're trying to accomplish - we want to find some neighboring particles (or photons, points in a cloud, etc) that fall within in a certain radius.  This can be done pretty easily if performance is not a problem.  You can simply loop over all of the particles and compare their distance with the radius.  If the distance is less than the radius, you have a neighbor!  For comparison purposes, we've included implementations of this algorithm in this project.

![](https://raw.githubusercontent.com/jeremynewlin/Accel/master/images/neighbor_example.png)
An example of a particle (green) and its neighbors (blue).

But, we ARE interested in performance, and the naive solution isn't going to cut it.  Enter the uniform grid, which is a type of spatial hashing data structure that allows for rapid nearest neighbor searches on the GPU (and CPU).

The basic idea of the structure can be described easily with a picture.

![](https://raw.githubusercontent.com/jeremynewlin/Accel/master/images/hash_grid_image.png)

In this example, we can particles [0,1,2,3,4,5] in the grid.  We hash the particles to their grid cell ids into an unsorted list.  Then, we sort that list by grid cell id.  Then, in the grid structure, we can simply store a single value - the index at which the grid cell start to contain particles in the sorted list.

So looking at Hash Cell 7 in the far right table, you can see that it has a value of 2.  This means that grid cell 7 contains particles starting at index 2 in the sorted list.  You can check to verify that this is indeed the case.  Then, the next Hash Cell that stores a value is number 10 with a value of 5.  You can then infer that Hash Cell 7 contains up to (but not including) index 5.  That way, we know what particles each grid cell contains.

Now, how does this help speed up nearest neighbor search?

Remember that before we simply looped over the particles to find our neighbors.  We're basically going to do the same thing here, but we've drastically reduced the number of particles we have to check.  If you carefully choose the grid cell size to be the same as the radius of your search, you only have to check the surrounding grid cells for potential neighbors.  By design any particle not within those cells will not pass the check of being less than the radius away from the current particle.

And now time for some performance evaluations.  First, we compared four different variations of the nearest neighbor search.  With/without the grid, and on the GPU/CPU.

First, here's a comparison of how particle number (and density) and neighbor amount affect performance.  Since that's a 3D data set, we got to make a 3D graph to show the results.  Videos can be found by clicking on the images below.

[![](https://raw.githubusercontent.com/jeremynewlin/Accel/master/images/reg.jpg)](https://www.youtube.com/watch?v=0zFS3fnT0FY&feature=youtu.be)

In the first video, you'll notice that the CPU brute force approach pops quite severely upwards as you increase both the neighbors and number of particles.  The other 3 stay much lower (you can't even see the GPU grid graph even).

[![](https://raw.githubusercontent.com/jeremynewlin/Accel/master/images/zoom.jpg)](https://www.youtube.com/watch?v=gPWu3BLeC0g&feature=youtu.be)

We zoom in on the data to look at the differences between the other 3.  You'll notice that zoomed in you can see that the CPU Grid and GPU Brute Force Approaches do in fact get larger, while the GPU Grid stays flatter.  Another interesting fact is that the CPU Grid sometimes outpaces the GPU Brute Force.

KD Tree
-----