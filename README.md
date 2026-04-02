# No more Docker usage for this project


# Tufts High Performance Computing Cluster (HPC) Setup and Workflow:
## Usage of Singularity for this project:
When we run our code on the HPC, we need to use Singularity (Docker for Computing Clusters) to run our container.
We accomplish this by doing the following:

### Pushing Docker image to remote repo:
1) Login to a docker hub account: `docker login`
2) Rename docker image to whatever you want it to be named in the repo: `docker tag <local_image>:<tag> <desired_name>:<desired_tag>`
3) Push to repo: `docker push <desired_name>:<desired_tag>`

### Pulling and Running Images on the HPC
1) Log into a HPC node
2) Log onto a compute node using: `srun -N1 -n8 -t0-8 -p preempt --gres=gpu:h100:1 --pty bash`
3) `module load singularity`
4) Use Singularity to pull, convert, and the docker image
   `singularity pull docker://<username>/<container_name>`
   If you just want a container that works with our code and don't need to make any other
   modifications, pull from `docker://aauretto122/skyblue_images:final-sp25`
5) This creates a .sif file that acts as a runnable image using the below steps:

### Helper scripts
There are two scripts on the HPC to make pulling and running code easier. They can be used like so: 
1) Log into HPC and get a terminal
2) Run `sh pull_image.sh <docker hub username>/<Container that trains model> --rm` 
   to get the latest version of the container that will train a model
3) Run `sh run_code <sif_file> <repo_root> [-it]`
   to execute the sif file specified with the path specified by repo_root mounted in. -it is optional; when present it will put the user in a shell inside the container and when absent `python /skyblue/src/model.py` is automatically executed inside the container. 

* Provision GPU on HPC
srun -N1 -n8 -t0-8 -p preempt --gres=gpu:h100:1 --pty bash

* HPC Cleanup
`singularity cache clean --force`
or
`rm -rf ~/.singularity`
Use this if the HPC is complaining about disk space (check quota with quota -s).

# Other Notes and Miscellaneous 
### PIREP GRID FORMAT ###
When the pireps are placed on their grid, the grid is `np.nan` everywhere else and NEG events are set to the background risk and will be later 0ed during spreading

### Satellite Data TimeStamp Notes

The AWS S3 Bucket from which GOES-16 ABI-L2 CMPIC Sattelite Images are pulled is not consistent with the TimeStamps matching when the satellite images were actually taken,
especially on earlier years such as 2017. The team believes that this is due to satellite images being post-hoc uploaded to the S3 Bucket. Therefore, the timestamps from which we expect to pull satellite data are generated using dt.datetime independently in our model pipeline to avoid any such inconsistency issues.