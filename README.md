# fly_re_identification

This code was used in the WACV publication: "Classification and Re-Identification of Fruit Fly Individuals Across Days using Convolutional Neural Networks"

Data can be downloaded from: https://doi.org/10.5683/SP2/JP4WDF

Example use-cases:

1. To train a resnet18 model on replicate-1 (present in pic_dir) using 'random cropping + masking' data augmentation, run:
```
python train_network.py --descriptor test --pic_dir /mnt/data/jschne02/Data/Nihal/3DayData --replicate 1 -d r_crop_mask --log_path /mnt/data/jschne02/Data/Nihal/logs --chkpoint_path /mnt/data/jschne02/Data/Nihal/checkpoints 
```
Note: The output is logged in the 'log_path' directory, and checkpoints are saved in the path given by 'chkpoint_path', and the name which is used in both cases is given by the 'descriptor' you specify

2. To test the above network on day-2 fly images of replicate-1, run:
```
python test_network.py --pic_dir /mnt/data/jschne02/Data/Nihal/3DayData --replicate 1 --day Day2 --chkpoint_path /mnt/data/jschne02/Data/Nihal/checkpoints --chkpoint_name test
```

3. To perform double day training (DDT) using 'random masking' data augmentation, run:
```
python train_network.py --descriptor test --pic_dir /mnt/data/jschne02/Data/Nihal/3DayData --replicate 1 --ddt true -d r_mask --log_path /mnt/data/jschne02/Data/Nihal/logs --chkpoint_path /mnt/data/jschne02/Data/Nihal/checkpoints 
```

4. To train domain adversarial networks on replicate-3 without any data augmentation, run:
```
python dann_train.py --descriptor test --pic_dir /mnt/data/jschne02/Data/Nihal/3DayData --replicate 3 --log_path /mnt/data/jschne02/Data/Nihal/logs --chkpoint_path /mnt/data/jschne02/Data/Nihal/checkpoints
```

5. To test the above trained domain adversarial network on Day-3 of replicate-3, run:
```
python dann_test.py --pic_dir /mnt/data/jschne02/Data/Nihal/3DayData --replicate 3 --day Day3 --chkpoint_path /mnt/data/jschne02/Data/Nihal/checkpoints --chkpoint_name test
```
