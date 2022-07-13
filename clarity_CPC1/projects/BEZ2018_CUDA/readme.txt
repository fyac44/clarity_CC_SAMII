%%% readme.txt for AN model %%%

This is the BEZ2018_CUDA version of the code for auditory periphery model 
from the Carney, Bruce and Zilany labs adapted to work with NVIDIA GPUs and
to compute the spike trains for SAMII.

This release implements the version of the model described in:

	Bruce, I.C., Erfani, Y., and Zilany, M.S.A. (2018). "A Phenomenological
	model of the synapse between the inner hair cell and auditory nerve: 
	Implications of limited neurotransmitter release sites," to appear in
	Hearing Research. (Special Issue on "Computational Models in Hearing".)

And also the adatations to compute SAMII in:

    Alvarez and Nogueira (2022). "Predicting Speech Intelligibility using
    the Spike Activity Mutual Information Index". INTERSPEECH 2022.

Please cite these papers if you publish any research results obtained with
this code or any modified versions of this code.

*** Change History ***

See the file changelog.txt (SAMII or CUDA related changes not here!!)

*** Instructions for Mex files ***

The Matlab and C code included with this distribution is designed to be
compiled as a Matlab MEX file, i.e., the compiled model MEX function will run
as if it were a Matlab function.  The code can be compiled within Matlab using
the function:

    mexANmodel.m

*** Instructions for CUDA files ***

In total, there are 2 kernels that must be compiled with nvidia toolkit to take advantage of the GPU.

It is important to make sure that the cuda toolkit is well installed. To check it type `nvcc --version` in the terminal. It should return the version of the cuda compiler.

More information can be found in https://developer.nvidia.com/cuda-toolkit

To compile type in the terminal:
nvcc -ptx model_IHC_BEZ2018.cu
nvcc -ptx model_Synapse_BEZ2018.cu

*** IMPORTANT NOTE ***
Original code for the BEZ2018 model can be found in:
https://www.ece.mcmaster.ca/~ibruce/zbcANmodel/zbcANmodel.htm
