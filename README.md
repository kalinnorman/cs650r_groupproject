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