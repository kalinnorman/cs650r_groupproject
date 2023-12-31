# CS 650R - Advanced Computer Vision Group Project
We propose implementing a custom structure from motion algorithm including custom rendering of the reconstructed geometry (either a dense 3D point cloud or a 3D mesh). The main steps that as part of this problem are:
1. Feature identification
2. Feature correspondence
3. Camera pose estimation
4. Triangulation
5. Sparse bundle adjustment (SBA)
6. Build dense point cloud
7. Rendering / visualization of point cloud or mesh

**Reminder: LOG HOURS**

## Task Divying Up:

Weekly touch base - **Tuesdays at 12pm on Discord**

- [ ] Dane - Camera Pose Estimation & Triangulation
- [ ] Kalin - GTSAM for Structure from Motion (steps 5 & 6)
- [X] Chad - Gather videos (simple objects)
- [X] Chad - Calibrate camera
- [X] Chad - Parse them into frames, get SIFT feature points, and do feature correspondence/matching
- [X] Chad - Find Open Source implementation of SfM (https://opensfm.org/docs/building.html) OR how to turn a sparse PC into a dense PC
