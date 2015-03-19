This code repository is dedicated to our new multi-label iterative relational neighbor classifier that employs social context features (SCRN). Our classifier incorporates a class propagation probability distribution obtained from instances’ social features, which are in turn extracted from the network topology. This class-propagation probability captures the node’s intrinsic likelihood of belonging to each class, and serves as a prior weight for each class when aggregating the neighbors’ class labels in the collective inference procedure. A relaxation labeling approach is used for the collective inference framework.

This material is based upon work supported by the National Science Foundation under Grant Number NSF IIS-08451 and the DARPA CSSG. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the funding agencies. 

Please cite our corresponding papers if you use this material in any form in your publication:

```
@inproceedings{Xi-Sukthankar-KDD2013,
    author = {Wang, Xi and Sukthankar, Gita},
    title = {Multi-Label Relational Neighbor Classification using Social Context Features},
    booktitle = {Proceedings of The 19th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
    address = {Chicago, USA},
    month = {Aug},
    year = {2013},
    pages = {464-472}
}
```
