To run a container in NV-A100 (mlt-clx129.sh.intel.com) machine, simply do  
```
docker run -it --privileged --gpus all \
	--shm-size 2G \
	-v /home2/pytorch-broad-models:/home2/pytorch-broad-models \
	-v /usr/local/cuda:/usr/local/cuda \
	--env HTTP_PROXY="http://child-prc.intel.com:913" \
	--env HTTPS_PROXY="http://child-prc.intel.com:913" \
	--env http_proxy="http://child-prc.intel.com:913" \
	--env https_proxy="http://child-prc.intel.com:913" \
	-p 913:913 oob-pt-train:v0.1
```
where the image name:tag is ```oob-pt-train:v0.1```