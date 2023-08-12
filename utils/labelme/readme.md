labelme data_annotated --labels labels.txt --nodata --validatelabel exact --config '{shift_auto_shape_color: -2}'
python labelme2voc.py data_annotated data_dataset_voc --labels labels.txt