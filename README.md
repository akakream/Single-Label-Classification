# Single Label Noisy Learning via Collaborative Training

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
keras\_model gives a decent accuracy if it is used with a batch\_size of 32, 64 or 128.
However, paper\_model gives a decent accuracy only with the batch\_size of 128. 

## How to prepare

Create the output folder:
```
mkdir output
```

## How to run

```
python dct.py -f co -a paper_model -d cifar10 -b 128 -e 6 -nt symmetry -nr 0.5 -dm mmd -si 100000.0 -sr 0.25 -lto 1.0 -ltr 1.0 
```

or just run the scipt in the scripts folder

## Empirical Results



