# Classification and Re-Identification of Fruit Fly Individuals Across Days with Convolutional Neural Networks

This code was used in the WACV publication: "Classification and Re-Identification of Fruit Fly Individuals Across Days with Convolutional Neural Networks" by Nihal Murali, [Jonathan Schneider](http://levinelab.com/team/jschneider), [Joel D. Levine](http://levinelab.com/team//jlevine), and [Graham W. Taylor](https://www.gwtaylor.ca).

To cite our work, please use the following bibtex entry.

```bibtex
@article{murali2019classification,
  Author = {Murali, Nihal and Schneider, Jonathan and Levine, Joel D. and Taylor, Graham W.},
  Title = {Classification and Re-Identification of Fruit Fly Individuals Across Days with Convolutional Neural Networks},
  Booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  Year = {2016}
}
```

Data can be downloaded from: https://doi.org/10.5683/SP2/JP4WDF

Example use-cases:

1. To train a resnet18 model on replicate-1 (present in pic_dir) using 'random cropping + masking' data augmentation, run:
```
python train_network.py --descriptor test --pic_dir </path/to/3DayData> --replicate 1 -d r_crop_mask --log_path </path/to/logs> --chkpoint_path </path/to/checkpoints> 
```
Note: The output is logged in the 'log_path' directory, and checkpoints are saved in the path given by 'chkpoint_path', and the name which is used in both cases is given by the 'descriptor' you specify

2. To test the above network on day-2 fly images of replicate-1, run:
```
python test_network.py --pic_dir </path/to/3DayData> --replicate 1 --day Day2 --chkpoint_path </path/to/checkpoints> --chkpoint_name test
```

3. To perform double day training (DDT) using 'random masking' data augmentation, run:
```
python train_network.py --descriptor test --pic_dir </path/to/3DayData> --replicate 1 --ddt true -d r_mask --log_path </path/to/logs> --chkpoint_path </path/to/checkpoints> 
```

4. To train domain adversarial networks on replicate-3 without any data augmentation, run:
```
python dann_train.py --descriptor test --pic_dir </path/to/3DayData> --replicate 3 --log_path </path/to/logs> --chkpoint_path </path/to/checkpoints>
```

5. To test the above trained domain adversarial network on Day-3 of replicate-3, run:
```
python dann_test.py --pic_dir </path/to/3DayData> --replicate 3 --day Day3 --chkpoint_path </path/to/checkpoints> --chkpoint_name test
```

You may be interested in our other related work, which makes use of the same dataset: [Can Drosophila melanogaster tell who’s who?](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0205043) by [Jonathan Schneider](http://levinelab.com/team/jschneider), Nihal Murali, [Graham W. Taylor](https://www.gwtaylor.ca), and [Joel D. Levine](http://levinelab.com/team//jlevine).
