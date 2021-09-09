# Jan's notes


## 2021/08/09

* Read through Tingwei's presentation slides
* Skimmed papers and changed titles
* Started literature review table on Google Sheets
* Started grabbing BibTex citation information

Many papers pick convolutional neural networks.
That seems like the obvious choice in architecture.
What about graph neural networks?
Could they help put the topology in topology optimization?

Questions to think about during literature review:

* What is the point of topology optimization?
* What are the challenges in topology optimization?
* What can machine learning techniques do for topology optimization?
* Which challenges, e.g. parameterization of topology, accelerating analysis, can they address?
* What are the recurring themes, i.e. convolutional neural networks, that you observe in this research at the intersection of machine learning and topology optimization?


## 2021/08/10

* Found and read through a new deal.II tutorial on topology optimization of elastic media
* Started GitHub repository for paper and, eventually, our experiments and results
* Started collecting BibTex citations
* Skimmed through DeepMind paper on graph neural networks for learning mesh-based simulation

[Link to deal.II tutorial](https://www.dealii.org/current/doxygen/deal.II/step_79.html)

[Link to DeepMind paper](https://sites.google.com/view/meshgraphnets)


## 2021/08/11

Read Tingwei's references, collected citations, and added notes to Google Sheet


## 2021/08/12

* Read Tingwei's references, collect citations, and added notes to Google Sheet
* Meet with Tingwei
* Looked into TopOpt group's PETSc topology optimization application

[Link to TopOpt PETSc app](https://www.topopt.mek.dtu.dk/Apps-and-software/Large-scale-topology-optimization-code-using-PETSc)

Takeaways from chat with Tingwei:

* Read Kallioras' "Accelerated..."
* Read Chi "Universal..."
* Read Hoyer "Neural reparameterization..."
* Build and run deal.II topology optimization tutorial


## 2021/08/13

* Talked to advisor about graph neural networks again and discovered new references (links below) 
* Tried build deal.II using Spack
  * using commit f0875b2fef8e495809854141785c1fc7f5d86407 because probably no tagged version has 9.3.0
  * ran into issue with Flex version so I specified version 2.6.4 in my `packages.yaml`
  * kept running out of memory because it was staging in `/tmp` so I configured it to default to something in the Spack directory
  * got stuck on LLVM (looks like the change above slowed things down considerably)

[Geometric Deep Learning: Going beyond Euclidean data](https://doi.org/10.1109/MSP.2017.2693418)

[Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478)


## 2021/08/16

Finally managed to build deal.II.
Had to go with the no-Gmsh option.


## 2021/08/17

Ran the deal.II topology optimization example and it worked.
Ask Tingwei what are our next steps.

* Do we try increasing the size of the problem? 
* What solver would we need to tackle larger scale problems? Iterative?
* What would we need to handle 3D?
* Do we try to integrate an external optimizer? Which one? IPOPT? MMA?
* How and at what point to integrate TensorFlow or PyTorch model of the density field?

Read Kallioras.
Not sure what's going on here.
Could Tingwei explain this?


## 2021/08/19

Takeaways from meeting with Tingwei:

* Kallioras' approach is similar to that in other papers
* Read "TONR..." (Zhang 2021)
* Read more papers and put them into the table in preparation for a literature review (Tingwei says Zhang has a good one to look at)
* Not a good idea to use something as complicated as deal.II for research platform
* Tingwei will try to put together a research platform based on TACS
* I will download the DeepMind graph neural net repo, try to reproduce their results and understand what they're doing


## 2021/08/24

I had some difficulty installing the dependencies in the MeshGraphNets directory.
I ignored the fact that the README created a virtual environment based on Python 3.6.
I used Python 3.8 instead because I already had it installed.
Then, when I went to install TensorFlow 1.15 using pip, pip threw an error saying that package wasn't found.
As it turns out, Python 3.8 was not supported when TensorFlow was on version 1.15.

These version issues may be a hassle in the future.
The DeepMind MeshGraphNet researchers also used the GPU version of TensorFlow 1.15, and I'm not sure if I can run it on my laptop, which does not have an Nvidia GPU.
We may have to write our own implementation of the graph nets based on a newer version of TensorFlow (or PyTorch), specifically one that does not require access to a GPU.

Still, I will attempt to get things running as they are, so that I can learn enough to be able to implement it elsewhere.


## 2021/08/25

According to the Python web site, support for Python 3.6 will end on 2021/12/23.
Even if I do get this up and running, we will have to update.

I installed Python 3.6 and created a virtual environment.
I upgraded pip and afterwards pip segfaulted every time I tried to install another package.

As it turns out, I wasn't able to install TensorFlow 1.15 using Python 3.6.
pip can only find up to TensorFlow 1.14.
I'm guessing that Python 3.7 might have been contemporary with TensorFlow 1.15.
I'll try that next.


## 2021/08/26

I installed Python 3.7, support for which ends on 2023-06-27.
I think we're good for a while then.
I tried to install the packages required and got the following error.

```
ERROR: tensorflow-gpu 1.15.5 has requirement numpy<1.19.0,>=1.16.0, but you'll have numpy 1.21.2 which is incompatible.
```

So I installed NumPy version 1.18.0 using the command

```
pip install numpy==1.18
```

I experimented with the `virtualenv` command in the MeshGraphNet README. 
I thought that it might be an easier way to get whatever version of Python you need for a project.
I thought it was a faster alternative to installing a version of Python from source and using the command

```
/path/to/python -m venv <environment-name>
```

but it isn't.
In the case of `virtualenv`, you still need to have the Python version for which you wish to create a virtual environment installed somewhere.
So you don't gain anything.

The `download_dataset.sh` script they provide requires the package `wget`.
I though that package was installed on my machine but it wasn't.
Something to keep in mind.

The way they're calling their Python code in the README file assumes the directory is treated as a module by Python.
So I had to add the directory above the `meshgraphnets` directory to my `PYTHONPATH` environment variable to get things running.
Once I got things running, the driver script spewed a bunch of output at me.
One of the things that came up was that it couldn't find the CUDA libraries on my laptop that doesn't have a GPU.
Go figure.
It still ran though.
But it's really fucking slow.

Given how slowly this is running, it is pretty clear to me that I'm not going to be able to reproduce their results on my machine.
How do I move forward?
I think there are two avenues, and I think I need to pursue both.

1. I will focus on learning what they are doing and how they are doing it. 
I need to understand what the graph neural network is and what it represents.
I need to understand how they are building it and training it.
This is not only an opportunity to learn concept but also to learn how the pros set up and run a machine learning project.
Ultimately, what I want to get out of this pursuit is sufficient knowledge to apply the graph neural network techniques to a smaller model problem of relevance to us.
2. I will explore how to run machine learning projects on Google Colab.
We currently don't have the means to run a machine learning project of significant scale.
I could try and set up my desktop, which has an AMD GPU, to work with TensorFlow or PyTorch, but that attempt would be time-consuming and might not succeed.
Setting up on Google Colab will present similar challenges but I think they will be less severe.
Even if I learn enough in part (1) to be able to reproduce most of the capability implemented in the `meshgraphnets` project with up-to-date modules, I still think it will be easier to work with Google Colab for our research than with our personal computing resources.


# 2021/08/30

Reading MeshGraphNets paper and taking notes


# 2021/08/31

Same as yesterday.
Don't know if they're using a GCN (graph convolutional network) as Tingwei read.
They contrast their architecture with another group that did use a GCN.


# 2021/09/02

What about cell complex neural networks?
Show [this link](https://towardsdatascience.com/one-network-to-rule-them-all-cell-complex-neural-networks-5920b4978a7c) to Tingwei.


# 2021/09/03

That article actually sucks.
How did so many spelling errors get past proofreading?
Here are two articles that discuss cell complex neural networks: [Weisfeiler and Lehman Go Cellular: CW Networks](https://arxiv.org/abs/2106.12575) and [Cell Complex Neural Networks](https://arxiv.org/abs/2010.00743).


# 2021/09/05

Talked to Tingwei about cell complex networks.
We agreed that we need to focus on a simple implementation, an example, using graph networks.
We can build from there to something more sophisticated.


## 2021/09/06

I've been looking at graph network libraries.
Here are my findings so far.

1. [Graph Nets Library](https://github.com/deepmind/graph_nets) from DeepMind.
It looks like it hasn't been worked on in some time, but it still may be worth looking into.
One of the co-authors of the MeshGraphNets paper appears to have written one of the examples, a spring-mass problem.
That might be the humble beginning we need.
2. [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric). 
It seems like we're leaning toward using TensorFlow, but, if that changes, here would be the reference implementations in PyTorch. 
3. [Deep Graph Library](https://github.com/dmlc/dgl) is supposedly independent of machine learning frameworks.
But it looks like it's not being actively developed. 
4. [Jraph](https://github.com/deepmind/jraph) is a weird one. 
Looks like nobody has added to it recently.
But I like it. 
It seems simple and it has potential for prototyping an implementation of cell complex neural networks.

A paper that we should probably read through as well is [this one](https://arxiv.org/abs/1806.01261).
It's the basis of a couple of libraries on this list.
The authors talk about unification just like the Deep Geometric Learning folks do.
Both groups talk about a zoo of architectures that they seem to want to unify under some universal taxonomy based on fundamental similarities and distinctions.

I also found a [playlist](https://geometricdeeplearning.com/lectures/) of Geometric Deep Learning lectures.


## 2021/09/07

Watching the video lectures
