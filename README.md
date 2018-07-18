# object_detection_server
object detection as a service



# config
1. set <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md' target='_blan' >pb</a> file path to ```PATH_TO_CKPT``` and pbtxt file path to ```PATH_TO_LABELS``` in ```odapi_server.py```
2. copy all file to ```tensorflow/models/research/```
 
 
# run 
```
python odapi.py
```


# visit
http://127.0.0.1:5000
