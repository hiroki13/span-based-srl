# A Span Selection Model for Semantic Role Labeling

## Citation
If you use our code, please cite our paper:

```
@inproceedings{ouchi:2018,
  title = {A Span Selection Model for Semantic Role Labeling},
}
```


## Prerequisites
* [python3](https://www.python.org/downloads/)
* [Theano](http://deeplearning.net/software/theano/)
* [h5py](https://www.h5py.org/)


## Data
### CoNLL-2005
* [Treebank-2](https://catalog.ldc.upenn.edu/LDC95T7)
### CoNLL-2012
* [OntoNotes Release 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
### Word Representations
* [SENNA](https://ronan.collobert.com/senna/download.html)
* [ELMo](https://github.com/allenai/allennlp/tree/v0.6.1)

### Data Format
#### CoNLL-2005
```
0:WORD 1:POS 2:PARSE 3:NE 4:FRAME-ID 5:LEMMA 6-:ARGS
Ms. NNP   (S1(S(NP*         *    -   -                    (A0*      
Haag NNP           *)    (LOC*)   -   -                       *)     
plays VBZ        (VP*         *    02  play                  (V*)     
Elianti NNP        (NP*))       *    -   -                    (A1*)     
. .             *))       *    -   -                       *      
```

#### CoNLL-2012
```
0:DOCUMENT 1:PART 2:WORD-INDEX 3:WORD 4:POS 5:PARSE 6:LEMMA 7:FRAME-ID 8:SENSE 9:SPEAKER 10:NE 11-N:ARGS N:COREF
bc/cctv/00/cctv_0001   0   0           This    DT  (TOP(S(NP*         -    -   -   Speaker#1        *   (ARG2*   (61
bc/cctv/00/cctv_0001   0   1            map    NN           *)        -    -   -   Speaker#1        *        *)   61)
bc/cctv/00/cctv_0001   0   2      reflected   VBD        (VP*    reflect  01   1   Speaker#1        *      (V*)    -
bc/cctv/00/cctv_0001   0   3            the    DT        (NP*         -    -   -   Speaker#1        *   (ARG1*     -
bc/cctv/00/cctv_0001   0   4       European    JJ           *         -    -   -   Speaker#1    (NORP)       *     -
bc/cctv/00/cctv_0001   0   5    battlefield    NN           *         -    -   -   Speaker#1        *        *     -
bc/cctv/00/cctv_0001   0   6      situation    NN          *))        -    -   -   Speaker#1        *        *)    -
bc/cctv/00/cctv_0001   0   7              .     .          *))        -    -   -   Speaker#1        *        *     -
```


## Usage
### Training and Validating
SENNA: `python src/main.py --mode train --train_data path/to/conll2005.train.txt --dev_data path/to/conll2005.dev.txt --data_type conll05 --drop_rate 0.1 --reg 0.0001 --hidden_dim 300 --n_layers 4 --halve_lr --word_emb path/to/senna --save --output_dir output`
ELMo: `python src/main.py --mode train --train_data path/to/conll2005.train.txt --dev_data path/to/conll2005.dev.txt --data_type conll05 --drop_rate 0.1 --reg 0.0001 --hidden_dim 300 --n_layers 4 --halve_lr --train_elmo_emb path/to/elmo.conll2005.train.hdf5 --dev_elmo_emb path/to/elmo.conll2005.dev.hdf5 --save --output_dir output`

### Predicting
SENNA: `python src/main.py --mode test --test_data path/to/conll2005.test.txt --data_type conll05 --drop_rate 0.1 --hidden_dim 300 --n_layers 4 --output_dir output --output_fn conll2005.test --word_emb path/to/senna --load_label output/label_ids.txt --load_param output/param.epoch-0.pkl.gz --search greedy`
ELMo: `python src/main.py --mode test --test_data path/to/conll2005.test.txt --data_type conll05 --drop_rate 0.1 --hidden_dim 300 --n_layers 4 --output_dir output --output_fn conll2005.test --test_elmo_emb path/to/elmo.conll2005.test.hdf5 --load_label output/label_ids.txt --load_param output/param.epoch-0.pkl.gz --search greedy`


## LICENSE
MIT License
