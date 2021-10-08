# GRIDSearch-Container 

## Introduction

> This container will load images specifically "PIXELDATA" from a dicom and match it to is given classification vector. Then Using a defult architecture it will build and train a model over a given set of gridsearch parameters. There are a multitude of parameters that the user is able to modify for generating the data set as well as the hyperparameter arrays over which the gridsearch is done. 

##  Design: 
  * Used python 
  * full list of packages needed: (listed within the Dockerfile.base)
    * pandas 
    * numpy 
    * matplotlib 
    * opencv-python 
    * python-math 
    * pydicom 
    * tensorflow 
    * scikit-learn 
    * tensorflow-addons 
    * pylibjpeg 
    * pylibjpeg-libjpeg 
    * python-gdcm 
    * tqdm 
    * keras 
    * imbalanced-learn 
   
##  How to use:
  > All the scripts are located within the "workspace" dir - any edits you will need to make for your specific use case will be with "gridsearch.py". Once edits are done run ./build.sh to build your docker container. Specifics to edit within docker are the Dockerfile.base file for naming the container, pushing to git and libraries used. If you want integration with XNAT navigate to the "xnat" folder and edit the command.json documentation available at @ https://wiki.xnat.org/container-service/making-your-docker-image-xnat-ready-122978887.html#MakingyourDockerImage%22XNATReady%22-installing-a-command-from-an-xnat-ready-image

## Running (ON XNAT): 
  * INCOMPLETE (NOT TESTED AS A CONTAINER ON XNAT)
  * Will work as just python script convert to jupyternotebook and run on there. 

## Running in general: 
  * Gridsearch.py load data from mortality.csv makes arrays for trining/testing/val and lunches a gridsearchCV
  * For my use cases i have dockersized it so I could run with access to GPU, your usecases may vary  
  * There are arguments needed to run this pipline which can be found within the gridsearch.py script 
  * there is an output file called params.txt which will be output at the end of the gridsearch with all results and best params found 

## NOTES: 
  * Until gridsearch is complete no output is generated, so keep that in mind when you decide how many different paramerters to tweak or how big the array is for any given parameter
  * Ideal use case is with a gpu to maximize performance, this runs 1 job at a time because I have not yet handeled the memory overflow error that crashes gridsearch when running multiple jobs at a time so use the "n_jobs=" parameter of gridsearch with caution  
  * This dockerzed to run for my use case it CANNOT RUN as an actual docker nore can it run as a XNAT Container 
  * Parts of the scripts within workspace were written with project specificity in mind so please keep that in mind as you use this code 
  * It is recommended that you have some experience working with docker and specficially building containers for xnat for this to work for your use cases 
  * If you just want to use the code for your own work without docker stuff just navigate to workspace copy the python files from it and edit them 
  
## Future: 
   * Arguments so that users can run this without manualy modifying code 
   * Potentially dockerizing it to be compatible with XNAT 
   * generalizing code even more 
