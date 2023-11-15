# cs650r_groupproject


## Installing OpenSfM

1. Clone OpenSfM
~~~
git clone --recursive https://github.com/mapillary/OpenSfM
cd OpenSfM
git submodule update --init --recursive
~~~

2. Create Conda Environment for OpenSfM

~~~
conda create -n sfm python=3
conda activate sfm
~~~

3. Edit `requirements.txt` changing the pyyaml to 5.3.1, then install python libraries

~~~
pip install -r requirements.txt
~~~

4. Build Docker container (for Ubuntu systems)

~~~
docker build -t <docker_container_name> .
~~~

5. Create and run Docker image

~~~
docker run --network="host" -it --rm <docker_container_name>
~~~

6. Build OpenSfM

~~~
python3 setup.py build
~~~

7. Not working properly (visit `https://opensfm.org/docs` instead) -> (Optional) To build documentation and browse locally at http://localhost:8000/
~~~
python3 setup.py build_doc
python3 -m http.server --directory build/doc/html/
~~~
If build error with no sphinx_rtd_theme, run `pip install sphinx-rtd-theme`.

## Using OpenSfM
`https://opensfm.org/docs/using.html`

1. Run SfM Docker Container
~~~
docker run --network="host" -it --rm <docker_container_name>
~~~

2. In a different terminal, find docker container id with the following command: `docker container ls`. Copy the id.

3. Place all dataset images in the `images` folder based on the following format `<dataset_name>/images/`. Example: `narnia_box/images/<img_names>.jpg`

4. In a different terminal, copy this dataset directory into the running docker container. Example: `sudo docker cp ../cs650r_groupproject/data/simple_objects/pumpkin 76aade637cfb:/source/OpenSfM/data/`
~~~
sudo docker cp /path/to/<dataset_name>/ <container_id>:/path/to/docker/container/data/
~~~

5. Copy `config.yaml` file from the Berlin data folder into your new data folder. Do this inside the Docker Container.

6. Run OpenSfM on data in the docker container
~~~
bin/opensfm_run_all data/<dataset_name>/
~~~

7. You can view the point cloud as discussed in `https://opensfm.org/docs/using.html`.

8. To view the dense mesh, copy it over to your host computer from the docker container. Example: `sudo docker cp 76aade637cfb:/source/OpenSfM/data/plant/undistorted/depthmaps/merged.ply /home/student/cs650r_groupproject/data/simple_objects/plant/merged_plant.ply`
~~~
sudo docker cp <container_id>:/path/to/docker/container/data/<dataset_name>/undistorted/depthmaps/merged.ply /path/to/cs650r_groupproject/data/simple_objects/<dataset_name>/merged_<dataset_name>.ply
~~~

9. Open up in `MeshLab` to view dense mesh


## Helpful `docker` commands

- Build Docker container
~~~
docker build -t <docker_container_name> .
~~~

- To run `localhost` while in the Docker container, start container with this command: 
~~~
docker run --network="host" -it --rm <docker_container_name>
~~~

- While container is running, find contained id by running the following command and copy it
~~~
docker container ls
~~~

- Copy files from container to host computer
~~~
sudo docker cp <container_id>:/container/file/location /host/file/location
~~~

<!-- - Setup Docker file syncing (doesn't allow me to run code with python errors)
~~~
docker run --network="host" -it -v /path/on/host:/path/in/container sfm
~~~ -->