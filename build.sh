cp Dockerfile.base Dockerfile && \
./command2label.py ./xnat/command.json >> Dockerfile && \
docker build -t xnat/model-gridsearch:latest .
docker tag xnat/model-gridsearch:latest registry.nrg.wustl.edu/docker/nrg-repo/yash/model-gridsearch:latest
rm Dockerfile
