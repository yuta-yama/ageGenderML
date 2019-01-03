# ageGenderML
Age Gender Deep Learning Test

## Training
The below training is based on code and instructions found at https://github.com/dpressel/rude-carnie

### Download Adience Dataset
We used pre-splits created by Levi and Hassner from the Adience Dataset. You can download the aligned face dataset and folds here:
http://www.openu.ac.il/home/hassner/Adience/data.html

To get the folds, you can git clone the below repository:

```
git clone https://github.com/GilLevi/AgeGenderDeepLearning
```

### Pre-process data for Training
First you will need to preprocess the data using preproc.py. This assumes that there is a directory that is passed for an absolute directory, as well as a file containing a list of the training data images and the label itself, and the validation data, and test data if applicable. The preproc.py program generates 'shards' for each of the datasets, each containing JPEG encoded RGB images of size 256x256

```
$ python preproc.py --fold_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_0 --train_list age_train.txt --valid_list age_val.txt --data_dir /data/xdata/age-gender/aligned --output_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0
```
The training (etc) lists are expected in the --fold_dir, and they contain first the relative path from the --data_dir and second the numeric label:

```
dpressel@dpressel:~/dev/work/3csi-rd/dpressel/sh$ head /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_0/age_train.txt
10069023@N00/landmark_aligned_face.1924.10335948845_0d22490234_o.jpg 5
7464014@N04/landmark_aligned_face.961.10109081873_8060c8b0a5_o.jpg 4
28754132@N06/landmark_aligned_face.608.11546494564_2ec3e89568_o.jpg 2
10543088@N02/landmark_aligned_face.662.10044788254_2091a56ec3_o.jpg 3
66870968@N06/landmark_aligned_face.1227.11326221064_32114bf26a_o.jpg 4
7464014@N04/landmark_aligned_face.963.10142314254_8e96a97459_o.jpg 4
113525713@N07/landmark_aligned_face.1016.11784555666_8d43b6c493_o.jpg 3
30872264@N00/landmark_aligned_face.603.9575166089_f5f9cecc8c_o.jpg 5
10897942@N03/landmark_aligned_face.633.10372582914_382144ffe8_o.jpg 3
10792106@N03/landmark_aligned_face.522.11039121906_b047c90cc1_o.jpg 3
```
Gender is done much the same way:
```
$ python preproc.py --fold_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_0 --train_list gender_train.txt --valid_list gender_val.txt --data_dir /data/xdata/age-gender/aligned --output_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0
```

### Train the model (Levi/Hassner)
Now that we have generated the training and validation shards, we can start training the program. Here is a simple way to call the driver program to run using SGD with momentum to train:
```
$ python train.py --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0
```
Once again, gender is done much the same way. Just be careful that you are running on the the preprocessed gender data, not the age data. Here we use a lower initial learning rate of 0.001
```
$ python train.py --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0 --max_steps 30000 --eta 0.001
```

### Train the model (fine-tuned Inception)
Its also easy to use this codebase to fine-tune an pre-trained inception checkpoint for age or gender dectection. Here is an example for how to do this:
```
$ python train.py --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0 --max_steps 15000 --model_type inception --batch_size 32 --eta 0.001 --dropout 0.5 --pre_model /data/pre-trained/inception_v3.ckpt
```
You can get the inception_v3.ckpt like so:
```
$ wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
```

### Monitoring the Training
You can easily monitor the job run by launching tensorboard with the --logdir specified in the program's initial output:
```
tensorboard --logdir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0/run-31376
```
Then navigate to http://127.0.0.1:6006/ in your browser to see results. The first tab (events) shows the loss over time, and the second shows the images that the network is seeing during training on batches.

### Evaluate the model
The evaluation program is written to be run alongside the training or after the fact. If you run it after the fact, you can specify a list of checkpoint steps to evaluate in sequence. If you run while training is working, it will periodically rerun itself on the latest checkpoint.

Here is an example of running evaluation continuously. The --run_id will live in the --train_dir (run-) and is the product of a single run of training (the id is actually the PID used in training):

```
$ python eval.py  --run_id 15918 --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0/ --eval_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/eval_gen_test_fold_is_0
```
Here is an after-the-fact run of eval that loops over the specified checkpoints and evaluates the performance on each:
```
$ python eval.py  --run_id 25079 --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0/ --eval_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/eval_age_test_fold_is_0 --requested_step_seq 7000,8000,9000,9999
```
To monitor the fine-tuning of an inception model, the call is much the same. Just be sure to pass --model_type inception
```
$ python eval.py  --run_id 8128 --train_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_0/ --eval_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/eval_age_test_fold_is_0 --model_type inception
```
