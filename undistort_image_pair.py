import json
import argparse
import numpy as np


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
    
    # Should only have 1 camera type
    assert(len(reconstruction_data["cameras"]) == 1)
    K = get_cam_intrinsics(list(reconstruction_data["cameras"].values())[0])
    print("Intrinsics Matrix:\n",K)

    # Access camera poses
    for shot_id, shot_data in reconstruction_data['shots'].items():
        # pose = shot_data['pose']
        print(f"Camera {shot_id} pose:")
        print("Rotation Matrix (axis-angle form):")
        print(shot_data['rotation'])
        print("Translation Vector:")
        print(shot_data['translation'])
        print("\n")
        break


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