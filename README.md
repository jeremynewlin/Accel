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

[![](https://raw.githubusercontent.com/jeremynewlin/Accel/master/images/reg.jpg)](https://www.youtube.com/watch?v=ay8P2ykL2Xk&feature=youtu.be)

In the first video, you'll notice that the CPU brute force approach pops quite severely upwards as you increase both the neighbors and number of particles.  The other 3 stay much lower (you can't even see the GPU grid graph even).

[![](https://raw.githubusercontent.com/jeremynewlin/Accel/master/images/zoom.jpg)](https://www.youtube.com/watch?v=HBxlWqiyqbc&feature=youtu.be)

We zoom in on the data to look at the differences between the other 3.  You'll notice that zoomed in you can see that the CPU Grid and GPU Brute Force Approaches do in fact get larger, while the GPU Grid stays flatter.  Another interesting fact is that the CPU Grid sometimes outpaces the GPU Brute Force.

First, let's take a look at the two methods on the GPU - using the grid and brute forcing it.

![](https://raw.githubusercontent.com/jeremynewlin/Accel/master/images/gpu_comp.png)

One interesting thing to note is that the the brute force stays relatively flat, while the grid is growing.  This could be a problem when the number of neighbors you need approaches the size of your data set, or if your data set is highly dense.  Our data set was relatively dense, and another important factor is sheer number of particles.  This was on the order of 10,000 particles, which really is not that much.  The brute force approach breaks down with many more particles (as seen in the videos above).

One other important metric we wanted to compare is how much overhead the grid is costing us.  Clearly we're getting a net benefit from the grid structure, but we thought it would be interesting to quantify how much overhead we're paying for the speedup.

![](https://raw.githubusercontent.com/jeremynewlin/Accel/master/images/gpu_brute_breakdown.png)

Clearly, in the brute force approach we're spending almost all of our time searching.  We don't need to set anything up (the memory management is done once, and thus is omitted from the comparison).

![](https://raw.githubusercontent.com/jeremynewlin/Accel/master/images/gpu_grid_breakdown.png)

However, in the grid approach, we're spending a lot of time not searching.  But that's a good thing!  The entire point of this structure is to reduce the search time.  The main costs here are the sorting step, and then the grid search.  The purple "other" represents mostly error checking.

Finally, we have to look at the memory overhead that comes with building this grid.  The cost of allocating the memory is amortized to 0, but we still need a sufficient amount of data to store the grid.  The extra amount of data is a function of the grid cell size, and will be discussed below.  The smaller the cell size, the greater the memory imprint.  Our grid is built around being unit sized, so your radius should be sufficiently small for most applications to prevent dampening or other loss of fine grained detail.

At simulations with lower particle count on the order of 1000, the memory imprint is significant in terms of percentage at lower radii:

![](https://raw.githubusercontent.com/jeremynewlin/Accel/master/images/memory_small_mesh.png)

However, it is important to note that this is still only on the order of 5 extra Mb in the worst case.

With higher particles counts, the extra memory becomes insignificant:

![](https://raw.githubusercontent.com/jeremynewlin/Accel/master/images/memory_big_mesh.png)

Overall, we're happy with grid - hopefully it will be of some use!

KD Tree
-----