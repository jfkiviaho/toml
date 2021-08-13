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
