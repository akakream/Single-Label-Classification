# Single-Label-Classification

Unofficial Tensorflow implementation of the paper [Learning from Noisy Labels via Discrepant Collaborative Training](http://openaccess.thecvf.com/content_WACV_2020/papers/Han_Learning_from_Noisy_Labels_via_Discrepant_Collaborative_Training_WACV_2020_paper.pdf)

> @inproceedings{han2020learning,
  title={Learning from Noisy Labels via Discrepant Collaborative Training},
    author={Han, Yan and ROY, SOUMAVA and Petersson, Lars and Harandi, Mehrtash},
      booktitle={The IEEE Winter Conference on Applications of Computer Vision},
        pages={3169--3178},
          year={2020}
}


At the time that I wrote this, there was no official implementations. There may be still not, not sure about that.

This code is not the exact implementation of the paper. There are minor differences here and there.

However the main concept is the same.

If you have any recommendation or offer of improvement, feel free to do a pull request! 

## Environment

* python 3.7.6
* tensorflow-gpu 2.2.0

## Caveats

There are two models that can be used paper\_model and keras\_model.
keras\_model gives a decent accuracy if it is used with a batch\_size of 32.
However, paper\_model gives a decent accuracy only with the batch\_size of 128. 

The implementation of the MMD module is not optimal and takes too much memory.
This code was ran on a Tesla P100-PCIE-16GB GPU and this gpu was not able to handle a batch\_size of 64 and 128.  

## How to run

python dct.py -f co -a paper\_model -d cifar10 -b 128 -e 6 

## Empirical Results



