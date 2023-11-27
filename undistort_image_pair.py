import os
import json
import argparse
import numpy as np

import cv2

'''
Steps taken to get reconstruction to use in this file:

1. Run SfM Docker Container
docker run --network="host" -it --rm <docker_container_name>

2. In a different terminal, find docker container id with the following command: docker container ls. Copy the id.

3. Place all dataset images in the images folder based on the following format <dataset_name>/images/. Example: narnia_box/images/<img_names>.jpg

4. In a different terminal, copy this dataset directory into the running docker container. sudo docker cp /path/to/<dataset_name>/ <container_id>:/path/to/docker/container/data/

5. Copy config.yaml file from the Berlin data folder into your new data folder. Do this inside the Docker Container.

6. Run OpenSfM on data in the docker container:
    6.1. /bin/opensfm extract_metadata data/<dataset_name>/
    6.2. /bin/opensfm detect_features data/<dataset_name>/
    6.3. /bin/opensfm match_features data/<dataset_name>/
    6.4. /bin/opensfm create_tracks data/<dataset_name>/
    6.5. /bin/opensfm reconstruct data/<dataset_name>/

7. Copy the reconstruction.json file over to your host computer from the docker container. 
sudo docker cp <container_id>:/path/to/docker/container/data/<dataset_name>/reconstruction.json /path/to/cs650r_groupproject/data/simple_objects/<dataset_name>/reconstruction_<dataset_name>.json
'''

'''
reconstruction.json: [RECONSTRUCTION, ...]

RECONSTRUCTION: {
    "cameras": {
        CAMERA_ID: CAMERA,
        ...
    },
    "shots": {
        SHOT_ID: SHOT,
        ...
    },
    "points": {
        POINT_ID: POINT,
        ...
    }
    "biases": {
        CAMERA_ID: {rot, trans, scale}
    }
    "rig_cameras": {
        CAMERA_ID: {rot, trans}
    }
    "rig_instances": {
        SHOT_ID: {trans,rot,rig_cam_ids}
    }
    "reference_lla": {lat,long,alt}
}

CAMERA: {
    "projection_type": "perspective",  # Can be perspective, brown, fisheye or equirectangular
    "width": NUMBER,                   # Image width in pixels
    "height": NUMBER,                  # Image height in pixels

    # Depending on the projection type more parameters are stored.
    # These are the parameters of the perspective camera.
    "focal": NUMBER,                   # Estimated focal length
    "k1": NUMBER,                      # Estimated distortion coefficient
    "k2": NUMBER,                      # Estimated distortion coefficient
}

SHOT: {
    "camera": CAMERA_ID,
    "rotation": [X, Y, Z],      # Estimated rotation as an angle-axis vector
    "translation": [X, Y, Z],   # Estimated translation
    "gps_position": [X, Y, Z],  # GPS coordinates in the reconstruction reference frame
    "gps_dop": METERS,          # GPS accuracy in meters
    "orientation": NUMBER,      # EXIF orientation tag (can be 1, 3, 6 or 8)
    "capture_time": SECONDS     # Capture time as a UNIX timestamp
}

POINT: {
    "coordinates": [X, Y, Z],      # Estimated position of the point
    "color": [R, G, B],            # Color of the point
}
'''

'''
Radial Distortion (what OpenSfM does):
x_dist = x ( 1 + k1 * r^2 + k2 * r^4 )
y_dist = y ( 1 + k1 * r^2 + k2 * r^4 )
'''

def get_cam_intrinsics(camera):
    f = camera["focal"]
    cx = camera["width"]//2
    cy = camera["height"]//2
    K = np.array([
        [f,0,cx],
        [0,f,cy],
        [0,0,1]
    ])
    return K

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_recon',type=str,help='/path/to/reconstruction.json')
    parser.add_argument('--left_img_name',type=str,default="l",help='left image name (as appears in reconstruction file')
    parser.add_argument('--right_img_name',type=str,default="r",help='right image name (as appears in reconstruction file')  
    return parser

if __name__ == '__main__':
    args = vars(read_args().parse_args())
    
    # # Get Camera Calibration params
    # cal_file = open(args['cam_cal'],'rb')

    # Load the reconstruction file
    with open(args['sfm_recon'], 'r') as file:
        reconstruction_data = json.load(file)[0]
    
    ## Get test image
    img_path = os.path.dirname(args['sfm_recon'])
    img_name = list(reconstruction_data["shots"].keys())[0]
    print("img_path:",img_path,"img_name:",img_name,"combined:",img_path + "/images/" + img_name)
    img = cv2.imread(img_path + "/images/" + img_name)

    ## Get distortion parameters
    # Should only have 1 camera type
    assert(len(reconstruction_data["cameras"]) == 1)
    camera_params = list(reconstruction_data["cameras"].values())[0]
    cx, cy = camera_params["width"]//2, camera_params["height"]//2
    K = get_cam_intrinsics(camera_params)
    print("Intrinsics Matrix:\n",K)
    dist_coeff = np.array([
        camera_params["k1"],
        camera_params["k2"],
        0,0,
    ]) # poss need two more zeros because 
    #   these distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements.


    ## Undistort Test Image
    h,  w = camera_params["height"], camera_params["width"]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, (w,h), 1, (w,h))
    # x, y, w, h = roi
    # undistorted_img = undistorted_img[y:y+h, x:x+w]
    print("New Cam Mat:",newcameramtx,"roi:",roi)
    
    undistorted_img = cv2.undistort(img, K, dist_coeff, None, newcameramtx)
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", img)
    cv2.namedWindow('Undistorted Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Undistorted Image", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Access camera poses
    # for shot_id, shot_data in reconstruction_data['shots'].items():
    #     # pose = shot_data['pose']
    #     print(f"Camera {shot_id} pose:")
    #     print("Rotation Matrix (axis-angle form):")
    #     print(shot_data['rotation'])
    #     print("Translation Vector:")
    #     print(shot_data['translation'])
    #     print("\n")
    #     break


    '''
    """Undistort an image into a set of undistorted ones.

    Args:
        shot: the distorted shot
        undistorted_shots: the set of undistorted shots covering the
            distorted shot field of view. That is 1 for most camera
            types and 6 for spherical cameras.
        original: the original distorted image array.
        interpolation: the opencv interpolation flag to use.
        max_size: maximum size of the undistorted image.
    """
    projection_type = shot.camera.projection_type
    if projection_type in ["perspective", "brown", "fisheye", "fisheye_opencv", "fisheye62"]:
        [undistorted_shot] = undistorted_shots
        new_camera = undistorted_shot.camera
        height, width = original.shape[:2]
        map1, map2 = pygeometry.compute_camera_mapping(
            shot.camera, new_camera, width, height
        )
        undistorted = cv2.remap(original, map1, map2, interpolation)
        return {undistorted_shot.id: scale_image(undistorted, max_size)}
    
    The c++ implementation in camera.cc of pygrometry.compute_camera_mapping():
    std::pair<MatXf, MatXf> ComputeCameraMapping(const Camera& from,
                                                const Camera& to, int width,
                                                int height) {
    const auto normalizer_factor = std::max(width, height);
    const auto inv_normalizer_factor = 1.0 / normalizer_factor;

    MatXf u_from(height, width);
    MatXf v_from(height, width);

    const auto half_width = width * 0.5;
    const auto half_height = height * 0.5;

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
        const auto uv = Vec2d(u - half_width, v - half_height);
        const Vec2d point_uv_from =
            normalizer_factor *
            from.Project(to.Bearing(inv_normalizer_factor * uv));
        u_from(v, u) = point_uv_from(0) + half_width;
        v_from(v, u) = point_uv_from(1) + half_height;
        }
    }
    return std::make_pair(u_from, v_from);
    }
    
    
    '''



    '''
    Undistorting an image:

    import numpy as np
    import cv2

    def undistort_image(distorted_image, k1, k2, p1, p2, k3):
        height, width = distorted_image.shape[:2]
        center = (width / 2, height / 2)

        # Create a grid of coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Convert coordinates to radial distance
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

        # Apply radial distortion correction
        r_corr = r * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)

        # Correct tangential distortion
        dx = 2 * p1 * x * y + p2 * (r**2 + 2 * x**2)
        dy = p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y

        # Apply distortion correction
        x_corr = x + dx
        y_corr = y + dy

        # Perform bilinear interpolation to get undistorted image
        x_floor = np.floor(x_corr).astype(int)
        y_floor = np.floor(y_corr).astype(int)

        x_frac = x_corr - x_floor
        y_frac = y_corr - y_floor

        x_floor = np.clip(x_floor, 0, width - 2)
        y_floor = np.clip(y_floor, 0, height - 2)

        undistorted_image = (
            (1 - x_frac) * (1 - y_frac) * distorted_image[y_floor, x_floor] +
            x_frac * (1 - y_frac) * distorted_image[y_floor, x_floor + 1] +
            (1 - x_frac) * y_frac * distorted_image[y_floor + 1, x_floor] +
            x_frac * y_frac * distorted_image[y_floor + 1, x_floor + 1]
        ).astype(np.uint8)

        return undistorted_image

    # Example usage
    distorted_image = ...  # Your distorted image
    undistorted_image = undistort_image(distorted_image, k1, k2, p1, p2, k3)

    # Display the results
    cv2.imshow('Distorted Image', distorted_image)
    cv2.imshow('Undistorted Image', undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''