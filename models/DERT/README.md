## Data Preparation

For Train:
```
  python data_preparation.py -c /Users/.../converted_train.csv -i True
```
For Test:
```
  python data_preparation.py -c /Users/.../converted_test.csv -i False
```
After that, [custom_train.json](/data/json_files/custom_train.json) and [custom_test.json](/data/json_files/custom_test.json) have been created in data folder as you will realize. 

## Training
After creation of these files, training can be started. 
```
  python train.py -n <train_image_folder_path> -t <test_image_folder_path>
```

__Also docker can be used for training__.  
If you want to use this project via docker (default image name --> detr):
```
  make docker
```
and then;
```
  make docker_run v=<full_path_of_the_project> n=<train_image_folder_path> t=<test_image_folder_path>
```

## Prediction
The objects in different images can be detected with model that it is created after training. 
```
  python prediction.py -p /Users/..../test/sample.jpg
```

