# Research Notes and Papers

Usefull Python Library: Open3D = New Python Library as of 2019 that is the software behind displaying 3D geometries

## My Densification Plan
- Test on two basic images:
- [x] Get camera poses and parameters (intrinsic and extrinsic) from OpenSfM
- [x] Radially Undistort left and right images
- [ ] Perform Image Rectification
  - [-] Estimate Essential matrix
  - [x] Decompose Essential matrix in $t$, $R$. OR Get Rotation & translation components from OpenSfM 
  - [x] Construct $R_{rect}$ from $t$ and $R$.
  - [x] Warp pixels in left and right images
- [ ] Compute Disparity map (Winner-Takes-All approach)
  - [ ] Choose disparity range based on point cloud in images
  - [ ] Do left-right consistency to remove outliers
- [ ] Get depth from disparity: $Z = \frac{f b}{d}$, where $f$ is the focal length, $b$ is the baseline between rectified images, and $d$ is the disparity value.
- [ ] (Poss) Add in global consistency method to encourage smoothness with Markov Random Fields (MRF)
- [ ] Run previous steps on all images
- [ ] Combine dense point clouds across all cameras, removing outliers/inconsistencies, to get a dense 3D point cloud of scene

## My surface mesh building plan
- [ ] Estimate normals of dense points
- [ ] Get a surface from the Point Cloud using Poisson Surface Reconstruction

## Densify the feature Point Cloud specifically from the SfM Algorithm
- **Stereo-Matching**: Potentially use "Pixelwise View Selection for Unstructured Multi-View Stereo" (ECCV 2016)
    - **Overview**: Stereo matching involves using two or more images of the same scene taken from slightly different viewpoints (stereo pairs). By comparing the disparities (horizontal pixel shifts) between corresponding points in these images, you can calculate the **<u>depth information for each pixel</u>**.
    - **Procedure**:
        1. Acquire stereo image pairs: Capture images of the scene from slightly different viewpoints. These images should overlap significantly.
        2. Compute stereo correspondence: Match features or pixels between the left and right images to find corresponding points.
        3. Disparity calculation: Calculate the disparity (horizontal shift) for each corresponding point.
        4. Depth calculation: Use the disparity information and the baseline distance between the cameras to calculate the depth for each pixel.
- **Use depth sensors** (e.g. LiDAR or Time-of-Flight Cameras)

## Densification of a Sparse Point Cloud

- TSDF (Truncated Signed Distance Field)
- CMVS/PMVS, Multi-View Environment (MVE), Shading-Aware Multi-View Stereo (SMVS)

## Or just jump straight to Differentiable Surface Splatting

- https://www.youtube.com/watch?v=MIu59GiJZ2s

## To get surfaces from Point clouds
- https://www.youtube.com/watch?v=C_WwL2mhxfw (do Poisson Surface Reconstruction)

## ColMap Approach
*Multi-View Stereo (MVS) takes the output of SfM to compute depth and/or normal information for every pixel in an image. Fusion of the depth and normal maps of multiple images in 3D then produces a dense point cloud of the scene. Using the depth and normal information of the fused point cloud, algorithms such as the (screened) Poisson surface reconstruction [kazhdan2013] can then recover the 3D surface geometry of the scene. More information on Multi-View Stereo in general and the algorithms in COLMAP can be found in [schoenberger16mvs].*

After reconstructing a sparse representation of the scene and the camera poses of the input images, MVS can now recover denser scene geometry. COLMAP has an integrated dense reconstruction pipeline to produce depth and normal maps for all registered images, to fuse the depth and normal maps into a dense point cloud with normal information, and to finally estimate a dense surface from the fused point cloud using Poisson [kazhdan2013] or Delaunay reconstruction.

To get started, import your sparse 3D model into COLMAP (or select the reconstructed model after finishing the previous sparse reconstruction steps). Then, choose Reconstruction > Multi-view stereo and select an empty or existing workspace folder, which is used for the output and of all dense reconstruction results. The first step is to undistort the images, second to compute the depth and normal maps using stereo, third to fuse the depth and normals maps to a point cloud, followed by a final, optional point cloud meshing step. During the stereo reconstruction process, the display might freeze due to heavy compute load and, if your GPU does not have enough memory, the reconstruction process might ungracefully crash. Please, refer to the FAQ (freeze and memory) for information on how to avoid these problems. Note that the reconstructed normals of the point cloud cannot be directly visualized in COLMAP, but e.g. in Meshlab by enabling Render > Show Normal/Curvature. Similarly, the reconstructed dense surface mesh model must be visualized with external software.

In addition to the internal dense reconstruction functionality, COLMAP exports to several other dense reconstruction libraries, such as CMVS/PMVS [furukawa10] or CMP-MVS [jancosek11]. Please choose Extras > Undistort images and select the appropriate format. The output folders contain the reconstruction and the undistorted images. In addition, the folders contain sample shell scripts to perform the dense reconstruction. To run PMVS2, execute the following commands:

./path/to/pmvs2 /path/to/undistortion/folder/pmvs/ option-all

where /path/to/undistortion/folder is the folder selected in the undistortion dialog. Make sure not to forget the trailing slash in /path/to/undistortion/folder/pmvs/ in the above command-line arguments.

For large datasets, you probably want to first run CMVS to cluster the scene into more manageable parts and then run COLMAP or PMVS2. Please, refer to the sample shell scripts in the undistortion output folder on how to run CMVS in combination with COLMAP or PMVS2. Moreover, there are a number of external libraries that support COLMAP’s output:

CMVS/PMVS [furukawa10]

CMP-MVS [jancosek11]

Line3D++ [hofer16].


(from https://colmap.github.io/tutorial.html)