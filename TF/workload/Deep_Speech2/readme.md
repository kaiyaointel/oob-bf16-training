local dataset : /lustre/dataset/OpenSLR_LibriSpeech_Corpus_dataset
0. download dataset from here http://www.openslr.org/12/
1. cd /path/to/models/research/deep_speech/
2. modify ./data/download.py 64L-> where the tar files
3. modify ./data/download.py184L-> where the path Decompression file storage 
4. bash generate_csv.sh
5. bash run_deep_speech.sh

you can also by below steps :
0. download dataset from here http://www.openslr.org/12/
1. git clone https://github.com/tensorflow/models.git
2. cd /models/research/deep_speech/
3. git apply run.patch
4. cp generate_csv.sh /models/research/deep_speech/ 
