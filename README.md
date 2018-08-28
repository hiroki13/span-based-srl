# span-based-srl

## Citation
If you use our code, please cite our paper:

```
@inproceedings{tan2018deep,
  title = {A Span Selection Model for Semantic Role Labeling},
}
```

## Usage
### Prerequisites
* python2
* A newer version of TensorFlow
* GloVe embeddings and srlconll scripts

### Data
#### CoNLL-2005
* [Treebank-2](https://catalog.ldc.upenn.edu/LDC95T7)
#### CoNLL-2012
* [OntoNotes Release 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
#### Word Representations
* [SENNA](https://ronan.collobert.com/senna/download.html)

### Training and Validating
Once you finished the procedures described above, you can start the training stage.
* Preparing the validation script

    An external validation script is required to enable the validation functionality.
    Here's the validation script we used to train an FFN model on the CoNLL-2005 dataset.
    Please make sure that the validation script can run properly.
```
SRLPATH=/PATH/TO/SRLCONLL
TAGGERPATH=/PATH/TO/TAGGER
DATAPATH=/PATH/TO/DATA

export PERL5LIB="$SRLPATH/lib:$PERL5LIB"
export PATH="$SRLPATH/bin:$PATH"

python $TAGGERPATH/main.py predict --data_path $DATAPATH/conll05.devel.txt \
  --model_dir train  --model_name deepatt \
  --vocab_path $DATAPATH/word_dict $DATAPATH/label_dict \
  --device_list 0 \
  --decoding_params="decode_batch_size=512" \
  --model_params="num_hidden_layers=10,feature_size=100,hidden_size=200,filter_size=800"
python $TAGGERPATH/scripts/convert_to_conll.py conll05.devel.txt.deepatt.decodes $DATAPATH/conll05.devel.props.gold.txt output
perl $SRLPATH/bin/srl-eval.pl $DATAPATH/conll05.devel.props.* output
```
* Training command

    The command below is what we used to train an model on the CoNLL-2005 dataset. The content of `run.sh` is described in the above section.
```
python tagger/main.py train \
    --data_path TRAIN_PATH --model_dir train --model_name deepatt \
    --vocab_path word_dict label_dict --emb_path glove.6B.100d.txt \
    --model_params=feature_size=100,hidden_size=200,filter_size=800,residual_dropout=0.2, \
                   num_hidden_layers=10,attention_dropout=0.1,relu_dropout=0.1 \
    --training_params=batch_size=4096,eval_batch_size=1024,optimizer=Adadelta,initializer=orthogonal, \
                      use_global_initializer=false,initializer_gain=1.0,train_steps=600000, \
                      learning_rate_decay=piecewise_constant,learning_rate_values=[1.0,0.5,0.25], \
                      learning_rate_boundaries=[400000,500000],device_list=[0],clip_grad_norm=1.0 \
    --validation_params=script=run.sh
```


### Decoding
The following is the command used to generate outputs:
```
python tagger/main.py predict \
    --data_path conll05.test.wsj.txt \
    --model_dir train/best --model_name deepatt \
    --vocab_path word_dict label_dict \
    --device_list 0 \
    --decoding_params="decode_batch_size=512" \
    --model_params="num_hidden_layers=10,feature_size=100,hidden_size=200,filter_size=800" \
    --emb_path glove.6B.100d.txt
```

### Model Ensemble
The command for model ensemble is similar to the one used in decoding:
```
python tagger/main.py ensemble \
    --data_path conll05.devel.txt \
    --checkpoints model_1/model.ckpt model_2/model.ckpt \
    --output_name output \
    --vocab_path word_dict1 word_dict2 label_dict \
    --model_params=feature_size=100,hidden_size=200,filter_size=800,num_hidden_layers=10 \
    --device_list 0 \
    --model_name deepatt \
    --emb_path glove.6B.100d.txt
```


### Pretrained Models
The pretrained models can be downloaded at [Google Drive](https://drive.google.com/open?id=1jvBlpOmqGdZEqnFrdWJkH1xHsGU2OjiP).

## Contact
This code is written by Zhixing Tan. If you have any problems, feel free to send an <a href="mailto:playinf@stu.xmu.edu.cn">email</a>.

## LICENSE
BSD
