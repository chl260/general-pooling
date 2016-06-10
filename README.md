<img align="left" src="http://pages.ucsd.edu/~chl260/fig/gpool.jpg" width="300">

#### Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree

We seek to improve deep neural networks by generalizing the pooling operations that play a central role in current architectures. We pursue a careful exploration of approaches to allow pooling to learn and to adapt to complex and variable patterns. The two primary directions lie in (1) learning a pooling function via combining of max and average pooling, and (2) learning a pooling function in the form of a tree-structured fusion of pooling filters that are themselves learned. In our experiments every generalized pooling operation we explore improves performance when used in place of average or max pooling. We experimentally demonstrate that the proposed pooling operations provide a boost in invariance properties relative to conventional pooling and set the state of the art on several widely adopted benchmark datasets. For detailed algorithm and experiment results please see our AISTATS 2016 [paper](http://arxiv.org/abs/1509.08985).

#### Demo: 
A quick demo of running the proposed pooling functions can be found at "models/generaling_pooling_AlexNet_example/". In this example, we adopt AlexNet model and simply replace the first max pooling with the proposed tree pooling (2 leaf nodes and 1 internal node) and replace the second and third max pooling with gated max-average pooling (1 gating mask each). After setting up the training and testing files, you can run the script "train_caffenet.sh" to start the training. Please also see "train_val.prototxt" file for the usage of the pooling layers and see "general_pooling.log" for the training process.

#### Transplant:
If you have different Caffe version than this repo and would like to try out the proposed pooling functions, you can go to "src/caffe/layers/" and transplant the following code to your repo using the instructions on this Caffe [Wiki](https://github.com/BVLC/caffe/wiki/Development) page to setup these layers.
- treepool_max_ave.cpp	(gated max-average pooling)
- treepool_max_ave.cu
- treepool_kernel_1layer.cpp (2 level tree pooling)
- treepool_kernel_1layer.cu	
- treepool_kernel_2layer.cpp (3 level tree pooling)	
- treepool_kernel_2layer.cu	

These files show a basic implementation of the proposed pooling layers. Further speed and memory optimization can be done using cuDNN library or some engineering; see the file "treepool_max_ave_efficient.c*", for an example.

Please cite the following paper if it helps your research:

    @inproceedings{lee2016generalizing,
      author = {Lee, Chen-Yu and Gallagher, Patrick and Tu, Zhuowen},
      booktitle = {International Conference on Artificial Intelligence and Statistics (AISTATS)},
      title = {Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree},
      year = {2016}
    }

#### Acknowledgment: 
This code is based on Caffe with new implemented pooling layers.

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year  = {2014},
    }

If you have any issues using the code please email me at chl260@ucsd.edu
