# Hours Spent Working on Project

| Date   | Hours |
| ------ | ----- |
| Oct 16 | 0.5   |
| Oct 21 | 3.0   |
| Oct 24 | 2.75  |
| Oct 25 | 0.25  |
| Oct 30 | 5.5   |
| Oct 31 | 3.5   |
| Nov 07 | 4.0   |
| Nov 15 | 4.0   |
| Nov 17 | 3.0   |
| Nov 20 | 2.0   |
| Nov 21 | 2.0   | Updated 2 pm

Total: 30.5 (Last updated on Nov 21 at 2 pm)


# Random Resources

This is just so that I have somewhere to put random resources I find while working on this project. 


## Open-source Implementations

- https://github.com/colmap/colmap -- From 2016, seems to be very highly regarded
- https://github.com/mapillary/OpenSfM
- https://github.com/borglab/gtsfm
- https://openmvg.readthedocs.io/en/latest/openMVG/sfm/sfm/


## General Overview of SfM / Tutorials

https://cmsc426.github.io/sfm/

- decent overview of all of the parts that go into structure from motion, but fairly high level (no code examples or things like that, just the math and some figures)

https://mi.eng.cam.ac.uk/~cipolla/publications/contributionToEditedBook/2008-SFM-chapters.pdf

- Chapter from a textbook that focuses on structure from motion. More detailed as it's from a textbook (40+ pages of material)

https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf

- "Structure-from-Motion Revisited", paper from 2016

https://link.springer.com/book/10.1007/978-3-642-24834-4

- "Structure from Motion using the Extended Kalman Filter", textbook from 2012


## Densify the Results

- https://www.di.ens.fr/willow/pdfs/cvpr07a.pdf - "Accurate, Dense, and Robust Multi-View Stereopsis"


## Perspective-n-Point

https://projet.liris.cnrs.fr/imagine/pub/proceedings/ECCV-2014/papers/8689/86890127.pdf

- "UPnP: An Optimal O(n) Solution to the Absolute Pose Problem with Universal Applicability", paper from 2014 that does PnP but also estimates essential matrix and so on (so this would be applicable when working with an uncalibrated camera)

## Bundle Adjustment

https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Triggs00.pdf

- "Bundle Adjustment — A Modern Synthesis", 75 page paper from 2000

https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

- Example of bundle adjustment using scipy (probably a decent starting place to see how this might be implemented)

https://dl.acm.org/doi/pdf/10.1145/1486525.1486527

- "SBA: A Software Package for Generic Sparse Bundle Adjustment", paper from 2009

## YouTube Videos that I watched

- https://www.youtube.com/watch?v=MyrVDUnaqUs -- 30 minute overview of bundle adjusment, overall it wasn't too technical, but it helped me learn more about COLMAP
- https://www.youtube.com/watch?v=lmj2Jk5tl60 -- Brief 5 minute explanation of bundle adjustment
