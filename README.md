# Automoated drone image alignment
A major issue I keep facing with drone data collected from agricultural field tirals (without RTK) is having to manually align orthos after the images have been stitched in webodm. There are methods such as SIFT and ORB but I havnt had very good luck with them as the orthos tend to be quite homogeneous, and can change a lot from one flight to the next, eg a herbicicde treatment may make some plots look very different over the course of a week, but may look very similar to the plot next door. When we do realignment manually this is fine, but the traditional methods that look more local extrema or similar fall down with the combination of spatial homogeneity and temporal heterogeneity.

Here we are going to attempt to build a Spatial Transformer Network to do most of the manual realignments for us. 

The training data is from multiple trials which were all flowen at least 4 different times - the trials look different every time and I have already manually aligned all of the flight orthos. 
The idea for training data is to randomly sample a pair of aligned images from a randomly selected trial, apply a constrained but random scale, rotation and shift to one of the images and have the model predict the transform - which we know because it is the inverse of the one we created to get the augmented image for the input. 

The final use case would take a base image in and an image to be aligned with the base image and the output would be the transform required to convert the target image to align with the base image. Note in this particular application we dont care about absolute positioning of the trail, so we just align everything to a base case, which may be out by a few meters due to the inacurcay of non-corrected gps data. 
