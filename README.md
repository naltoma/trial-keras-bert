## summary

<hr>

## whatis
新システムとしてTesla V100SでSingularityでSlurmな環境が導入されたので、[BERT with SentencePiece を日本語 Wikipedia で学習してモデルを公開しました](https://yoheikikuta.github.io/bert-japanese/)を使って[livedoor ニュースコーパス](https://www.rondhuit.com/download.html)のジャンル推定するfine-tuningしてみた。

## requirements
- Singularity
- Slurm
- Pipfile参照。
    - keras-bert, tensorflow(**2.x**), keras, sentencepiece, scikit-learn, transformers

## howto
### 1. build a singularity container image
下記手順で生成したコンテナイメージ（test.sif）を適当な場所に保存。

```shell
singularity build --fakeroot --sandbox test docker://tensorflow/tensorflow:latest-gpu-py3
singularity shell --fakeroot --writable test
pip install keras-bert tensorflow keras sentencepiece scikit-learn transformers
exit
singularity build --fakeroot test.sif test
```

### 2. ready for pre-trained models
- [BERT with SentencePiece を日本語 Wikipedia で学習してモデルを公開しました](https://yoheikikuta.github.io/bert-japanese/)からダウンロード。
    - wiki-ja.*: SentencePieceの学習済みモデル。
    - model.ckpt-1400000.*: SentencePiece+BERTで学習済みモデル。
    - これらを ``~/model/bert-wiki-ja/`` に配置すると仮定。もしくは [train.py](./train.py) の ``pretrained_path`` を置いた場所に修正しよう。
- [bert_config.json](./bert_config.json) を準備。
    - ここでは前述の学習済みモデルと同じディレクトリに保存すると仮定。

### 3. ready for dataset
[livedoor ニュースコーパス](https://www.rondhuit.com/download.html)から ``ldcc-20140209.tar.gz`` をダウンロードし、展開。これを ``~/data/livedoor-text/`` に配置すると仮定。もしくは [train.py](./train.py) の ``dataset_path`` を置いた場所に修正しよう。

### 4. ready for sources
- ここのクローンを用意。
- fine-tuning したモデルは ``pretrained_path/model_livedoor.h5`` として保存します。変更したい場合には ``model_path`` を修正してください。

### 5. ready for a batch file for Slurm
- [train.sbatch](./train.sbatch) のようにバッチファイルを準備。SIFイメージやtrain.pyを置く場所を変更しているなら修正しよう。

### 6. run 
Slurmにジョブを投入。実行ログは ``logs/`` に保存されます。

```shell
sbatch train.sbatch
```

<hr>

## notes
- データセット上の処理。
    - [livedoor ニュースコーパス](https://www.rondhuit.com/download.html)に含まれる本文は、冒頭1行目がURL、2行目がタイムスタンプです。これらはジャンル推定に全く寄与しないか、もしくはURLから判定しやすくなる可能性があります。このため冒頭2行は除外し、3行目から利用しています。なお、3行目はタイトルでありかなり重要な情報が含まれているため、これも除外する人もいますが、今回は含めたままにしています。
- 利用している学習済みモデルに伴う制約。
    - [BERT with SentencePiece を日本語 Wikipedia で学習してモデルを公開しました](https://yoheikikuta.github.io/bert-japanese/)は、最大文字数（max_position_embeddings）が512です。これに制御文字としての``[CLS]``, ``[SEP]``の2文字分を確保した510文字がBERTに入力可能な最大文字数です。前述の通りlivedoorコーパスは3行目から
- BERTで用いているTensorFlow, tf.keras, Keras, PyTorch,,等のバージョンに注意。
    - 例えば[初出のBERT](https://github.com/google-research/bert)はTensorFlow 1.xでのみ動作。なお、TensrFlow 2.x では[contribuを除けばmigrationで対応できる](https://www.tensorflow.org/guide/migrate)とあるが、BERTはcontribを用いているためこれでは対応しきれない。公式パッチ作るつもりで暫く取り組んでみたがあまりにもエラー頻出するため断念した。
- tf.keras なのか Keras なのか keras-bert なのか問題。
    - GitHub上の初出BERTとは別に、BERT実装は豊富に提供されている。されすぎているかもしれない。これらが混在しているため、keras利用する際にはどれを使っているのか確認しよう。
- GPU使う場合
    - Singularityでコンテナイメージを作成するなら、利用するパッケージのGPU版イメージをダウンロードし、そこからsandbox作成してカスタマイズする流れが恐らく最もスムーズ。
        - 当初全てのパッケージを含むコンテナを作ろうと試行錯誤していた。ぶっちゃけそっちの方に圧倒的に時間かけたのだけど無駄足だった感。。PyTorchはconda、TensorFlowはpip（とbazel）のため[混ぜないほうがベター](http://onoz000.hatenablog.com/entry/2018/02/11/142347)。
        - なお、GPU版パッケージが動作しているかどうかを確認するには ``singularity shell --nv`` で起動した状態で確認しよう。

```Python
# Tensorflow の動作確認
import tensorflow as tf
print(tf.test.is_gpu_available)
print(tf.test.gpu_device_name()) # => GPU (device: 0, name: Tesla V100S-PCIE-32GB, pci bus id: 0000:3b:00.0, compute capability: 7.0)

# PyTorchの動作確認
import torch
print(torch.cuda.is_available()) #=> True
print(torch.version.cuda) # => 11.0
print(torch.backends.cudnn.enabled) # => 11.0
```

<hr>

## others
### 学習中のGPU利用状況
ミニバッチ16とそれほど大きくないですが、結構メモリ食ってる？

```
amane:tnal% nvidia-smi
Wed Dec  2 19:58:57 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.80.02    Driver Version: 450.80.02    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100S-PCI...  Off  | 00000000:3B:00.0 Off |                    0 |
| N/A   58C    P0   122W / 250W |  31027MiB / 32510MiB |     74%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   3367442      C   /usr/local/bin/python           31023MiB |
+-----------------------------------------------------------------------------+
```

### 実行結果
- GPU1個使った場合の例。
    - 学習済みモデルからベクトル生成する部分でもかなり時間かかってます。7376文書で741秒なので約10文書/1秒。
    - ``EPOCHS=10, BATCH_SIZE=16`` でfine-tuningに266秒。約26秒/1エポック。
    - モデル構築等含めた全体の処理時間は17分20秒。

```
Wed Dec  2 19:45:27 JST 2020
Model: "functional_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Input-Token (InputLayer)        [(None, 512)]        0                                            
__________________________________________________________________________________________________
Input-Segment (InputLayer)      [(None, 512)]        0                                            
__________________________________________________________________________________________________
Embedding-Token (TokenEmbedding [(None, 512, 768), ( 24576000    Input-Token[0][0]                
__________________________________________________________________________________________________
Embedding-Segment (Embedding)   (None, 512, 768)     1536        Input-Segment[0][0]              
__________________________________________________________________________________________________
Embedding-Token-Segment (Add)   (None, 512, 768)     0           Embedding-Token[0][0]            
                                                                 Embedding-Segment[0][0]          
__________________________________________________________________________________________________
Embedding-Position (PositionEmb (None, 512, 768)     393216      Embedding-Token-Segment[0][0]    
__________________________________________________________________________________________________
Embedding-Dropout (Dropout)     (None, 512, 768)     0           Embedding-Position[0][0]         
__________________________________________________________________________________________________
Embedding-Norm (LayerNormalizat (None, 512, 768)     1536        Embedding-Dropout[0][0]          
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, None, 768)    2362368     Embedding-Norm[0][0]             
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, None, 768)    0           Encoder-1-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, 512, 768)     0           Embedding-Norm[0][0]             
                                                                 Encoder-1-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-1-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-1-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-1-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-1-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-1-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-1-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-1-MultiHeadSelfAttention-
                                                                 Encoder-1-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-1-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-1-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, None, 768)    2362368     Encoder-1-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, None, 768)    0           Encoder-2-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-1-FeedForward-Norm[0][0] 
                                                                 Encoder-2-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-2-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-2-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-2-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-2-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-2-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-2-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-2-MultiHeadSelfAttention-
                                                                 Encoder-2-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-2-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-2-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, None, 768)    2362368     Encoder-2-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, None, 768)    0           Encoder-3-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-2-FeedForward-Norm[0][0] 
                                                                 Encoder-3-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-3-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-3-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-3-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-3-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-3-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-3-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-3-MultiHeadSelfAttention-
                                                                 Encoder-3-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-3-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-3-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, None, 768)    2362368     Encoder-3-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, None, 768)    0           Encoder-4-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-3-FeedForward-Norm[0][0] 
                                                                 Encoder-4-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-4-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-4-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-4-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-4-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-4-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-4-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-4-MultiHeadSelfAttention-
                                                                 Encoder-4-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-4-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-4-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, None, 768)    2362368     Encoder-4-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, None, 768)    0           Encoder-5-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-4-FeedForward-Norm[0][0] 
                                                                 Encoder-5-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-5-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-5-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-5-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-5-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-5-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-5-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-5-MultiHeadSelfAttention-
                                                                 Encoder-5-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-5-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-5-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, None, 768)    2362368     Encoder-5-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, None, 768)    0           Encoder-6-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-5-FeedForward-Norm[0][0] 
                                                                 Encoder-6-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-6-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-6-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-6-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-6-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-6-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-6-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-6-MultiHeadSelfAttention-
                                                                 Encoder-6-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-6-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-6-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, None, 768)    2362368     Encoder-6-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, None, 768)    0           Encoder-7-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-6-FeedForward-Norm[0][0] 
                                                                 Encoder-7-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-7-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-7-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-7-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-7-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-7-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-7-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-7-MultiHeadSelfAttention-
                                                                 Encoder-7-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-7-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-7-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, None, 768)    2362368     Encoder-7-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, None, 768)    0           Encoder-8-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-7-FeedForward-Norm[0][0] 
                                                                 Encoder-8-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-8-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-8-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-8-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-8-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-8-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-8-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-8-MultiHeadSelfAttention-
                                                                 Encoder-8-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-8-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-8-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, None, 768)    2362368     Encoder-8-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, None, 768)    0           Encoder-9-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-8-FeedForward-Norm[0][0] 
                                                                 Encoder-9-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-9-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-9-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-9-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-9-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-9-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-9-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-9-MultiHeadSelfAttention-
                                                                 Encoder-9-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-9-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-9-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, None, 768)    2362368     Encoder-9-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, None, 768)    0           Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, 512, 768)     0           Encoder-9-FeedForward-Norm[0][0] 
                                                                 Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, 512, 768)     1536        Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-FeedForward (FeedFor (None, 512, 768)     4722432     Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-FeedForward-Dropout  (None, 512, 768)     0           Encoder-10-FeedForward[0][0]     
__________________________________________________________________________________________________
Encoder-10-FeedForward-Add (Add (None, 512, 768)     0           Encoder-10-MultiHeadSelfAttention
                                                                 Encoder-10-FeedForward-Dropout[0]
__________________________________________________________________________________________________
Encoder-10-FeedForward-Norm (La (None, 512, 768)     1536        Encoder-10-FeedForward-Add[0][0] 
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, None, 768)    2362368     Encoder-10-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, None, 768)    0           Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, 512, 768)     0           Encoder-10-FeedForward-Norm[0][0]
                                                                 Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, 512, 768)     1536        Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-FeedForward (FeedFor (None, 512, 768)     4722432     Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-FeedForward-Dropout  (None, 512, 768)     0           Encoder-11-FeedForward[0][0]     
__________________________________________________________________________________________________
Encoder-11-FeedForward-Add (Add (None, 512, 768)     0           Encoder-11-MultiHeadSelfAttention
                                                                 Encoder-11-FeedForward-Dropout[0]
__________________________________________________________________________________________________
Encoder-11-FeedForward-Norm (La (None, 512, 768)     1536        Encoder-11-FeedForward-Add[0][0] 
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, None, 768)    2362368     Encoder-11-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, None, 768)    0           Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, 512, 768)     0           Encoder-11-FeedForward-Norm[0][0]
                                                                 Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, 512, 768)     1536        Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-FeedForward (FeedFor (None, 512, 768)     4722432     Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-FeedForward-Dropout  (None, 512, 768)     0           Encoder-12-FeedForward[0][0]     
__________________________________________________________________________________________________
Encoder-12-FeedForward-Add (Add (None, 512, 768)     0           Encoder-12-MultiHeadSelfAttention
                                                                 Encoder-12-FeedForward-Dropout[0]
__________________________________________________________________________________________________
Encoder-12-FeedForward-Norm (La (None, 512, 768)     1536        Encoder-12-FeedForward-Add[0][0] 
==================================================================================================
Total params: 110,026,752
Trainable params: 0
Non-trainable params: 110,026,752
__________________________________________________________________________________________________
{'it-life-hack': 0, 'peachy': 1, 'smax': 2, 'livedoor-homme': 3, 'sports-watch': 4, 'topic-news': 5, 'dokujo-tsushin': 6, 'kaden-channel': 7, 'movie-enter': 8}
_text_to_vectors(): begin.
_text_to_vectors(): 1000 done...
_text_to_vectors(): 2000 done...
_text_to_vectors(): 3000 done...
_text_to_vectors(): 4000 done...
_text_to_vectors(): 5000 done...
_text_to_vectors(): 6000 done...
_text_to_vectors(): 7000 done...
_text_to_vectors(): 7376 done.
_text_to_vectors(): time =  741.18 sec
len(x_train), len(y_train):  5900 5900
model.fit(): begin
Epoch 1/10
369/369 [==============================] - 25s 69ms/step - loss: 0.5870 - mae: 0.0642 - mse: 0.0310 - acc: 0.8039
Epoch 2/10
369/369 [==============================] - 26s 69ms/step - loss: 0.2039 - mae: 0.0246 - mse: 0.0110 - acc: 0.9375
Epoch 3/10
369/369 [==============================] - 26s 69ms/step - loss: 0.0848 - mae: 0.0114 - mse: 0.0046 - acc: 0.9724
Epoch 4/10
369/369 [==============================] - 26s 69ms/step - loss: 0.0470 - mae: 0.0067 - mse: 0.0026 - acc: 0.9847
Epoch 5/10
369/369 [==============================] - 25s 69ms/step - loss: 0.0239 - mae: 0.0035 - mse: 0.0012 - acc: 0.9922
Epoch 6/10
369/369 [==============================] - 25s 69ms/step - loss: 0.0142 - mae: 0.0019 - mse: 6.3406e-04 - acc: 0.9968
Epoch 7/10
369/369 [==============================] - 25s 69ms/step - loss: 0.0066 - mae: 8.2529e-04 - mse: 2.6925e-04 - acc: 0.9983
Epoch 8/10
369/369 [==============================] - 25s 69ms/step - loss: 0.0193 - mae: 0.0025 - mse: 9.7851e-04 - acc: 0.9939   
Epoch 9/10
369/369 [==============================] - 25s 69ms/step - loss: 0.0465 - mae: 0.0059 - mse: 0.0025 - acc: 0.9842
Epoch 10/10
369/369 [==============================] - 25s 69ms/step - loss: 0.0268 - mae: 0.0035 - mse: 0.0014 - acc: 0.9922
model.fit(): time =  266.24 sec
model.predict(): begin
47/47 [==============================] - 2s 50ms/step
model.predict(): time =  3.77 sec
ACC:  0.8868563685636857
                precision    recall  f1-score   support

  it-life-hack       0.87      0.91      0.89       167
        peachy       0.93      0.70      0.80       185
          smax       0.97      0.90      0.93       193
livedoor-homme       0.85      0.70      0.76       105
  sports-watch       0.97      0.95      0.96       178
    topic-news       0.89      0.97      0.92       144
dokujo-tsushin       0.93      0.86      0.89       160
 kaden-channel       0.77      0.98      0.86       165
   movie-enter       0.85      0.98      0.91       179

      accuracy                           0.89      1476
     macro avg       0.89      0.88      0.88      1476
  weighted avg       0.89      0.89      0.89      1476

Wed Dec  2 20:02:46 JST 2020
```

<hr>

## todo?
- refactoring

## referencese
- [keras-bert](https://github.com/CyberZHG/keras-bert)
    - [classification demo](https://github.com/CyberZHG/keras-bert/blob/master/demo/tune/keras_bert_classification_tpu.ipynb)
    - [SentencePiece + 日本語WikipediaのBERTモデルをKeras BERTで利用する](https://www.inoue-kobo.com/ai_ml/keras-bert/index.html)
