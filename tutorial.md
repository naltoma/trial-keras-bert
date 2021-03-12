# ハンズオン: 新システムの Singularity + Slurm 環境下でGPUを使ってみよう
- ＜目次＞
    - 前提
    - ゴール
    - ハンズオンの流れ
    - 全体像
    - Tips
        - (a) どうやってコンテナを用意するのか
        - (b) どうやってSlurmで動かすのか
        - その他

---
## 前提
- [Singularityのすゝめ](https://ie.u-ryukyu.ac.jp/syskan/service/singularity/)
- amaneにログインできる。（できない人はシス管に相談しよう）

---
## ゴール
- （自分に）必要なコンテナイメージを作成できる。
- コンテナイメージで実行したプログラムが、実際にGPUで処理されていることを確認できる。
- Slurmへのジョブ投入を通してプログラムを実行できる。
- 別の機会があればやるかも。
    - (学習したモデルを保存し、復元して利用できる。)
    - (計測して実行時間目安を見積もれるようになる。)

---
## ハンズオンの流れ
取り敢えず下記の step 4 までやろう。コンテナ作成に10分ぐらい時間かかる予定なので、作成中に解説をはさみます。

```shell
# step 1: amaneにログイン。作業用ディレクトリを作成して移動。
ssh amane
mkdir temp
cd temp

# step 2: リポジトリのクローンを用意。
git clone https://github.com/naltoma/trial-keras-bert.git
cd trial-keras-bert

# step 3: 実行に必要なモデルとデータセットのリンクを用意。
ln -s ~tnal/model ~/model
ln -s ~tnal/data ~/data

# step 4: コンテナの用意。
singularity pull docker://tensorflow/tensorflow:latest-gpu-py3
singularity build --fakeroot trial.sif keras-bert.def

# step 5: Singularityで実行。
# step5,6は動作確認をするためのお試し。
# 本来は直接実行せず、ジョブスケジューラSlurmに任せる（step7,8参照）。
singularity exec --nv trial.sif python train.py

# step 6: **別端末** からamaneにログインして動作確認。
# ここで Processes 欄に実行したプロセスが記載されていること。
# 動作確認できたら step 5 のプロセスは強制終了しよう。
nvidia-smi

# step 7: Slurm経由で実行するためにバッチファイルを作成。
# ベースがあるので、コンテナイメージを今回作成したものに修正しよう。
vim train.sbatch

# step 8: Slurm経由で実行。
sbatch train.sbatch

# step 9: ジョブ確認。
# squeueで NAME, JOBID を確認。
squeue

# step 10: tail による動作確認。
# 今回用意した train.sbatch では「logs/%x-%j.log」に標準出力を書き出すよう指定している。
# %xはNAME, %jはJOBIDに置き換えられる。
# 下記の xxx は step 9 で確認した JOBID を指定しよう。
# 確認済んだら Ctrl-c でtailコマンドを終了。
tail -f keras-bert-normal-xxx.log

# step 11: 後片付け
cd ..
rm -rf trial-keras-bert
rm ~/model
rm ~/data
```

---
## 全体像
- [trial-keras-bert](https://github.com/naltoma/trial-keras-bert)は自然言語処理を深層学習でやる例。処理内容に興味がある人は[補足](footnotes.md)を参照しよう。今回のハンズオン的には以下の流れを意識しよう。
    - GPUで動かしたいプログラムがあるとする。（今回はこのリポジトリのtrain.py。上記step 1〜3）
    - そのプログラムに必要な環境をコンテナとして用意する。（上記step 4）
    - 用意したコンテナを直接動かすのではなく、Slurm経由で動かす。（上記step 8）
- 今回のハンズオン主題は **(a) どうやってコンテナを用意するのか**、**(b) どうやってSlurmで動かすのか** の2点。
- Singularity, Slurmの概説は[footnotes.md](footnotes.md)参照。

---
## Tips
### (a) どうやってコンテナを用意するのか
- step a-1: パッケージ管理ツールを明確にする。
    - pip と conda どちらで環境構築するかを決める。
        - pip, conda はどちらもPythonパッケージ管理ツール。管理方法が違う割には同じ場所で管理しようとするため[混ぜて使うのは避けたほうが良い](http://onoz000.hatenablog.com/entry/2018/02/11/142347)。一応改善傾向にはあるらしいが、避けたほうが良いのは相変わらずの模様。
        - 今回は pip だけ使うことに。
- step a-2: パッケージを洗い出す。
    - pip freeze は避けたほうが良い。
        - Pythonでpipを使っているなら ``pip freeze > requirements.txt`` で使っているパッケージ一覧をバージョンと一緒に出力できる。GPU使わない環境なら ``pip install -r reqquirements.txt`` で環境再現できる。しかし、無駄が多いのと、GPU使う場合はこれではうまくいかないことが多いので避けよう。
        - 無駄というのは、pip freeze では「直接は利用していない関連パッケージ」まで全てバージョン指定で列挙されてしまうから。実際には直接利用しているパッケージを適切に指定してやれば、後はpipがよしなにやってくれる。だから pip freeze は避けたほうが良い。
        - GPU使う場合にうまくいかないというのは、例えば [TensorFlow公式のGPU対応版Dockerイメージ](https://hub.docker.com/r/tensorflow/tensorflow/) は、2021/3/7時点で ``Python 3.6.9, tensorflow-gpu==2.1.0``。これに対してGPU無視して pip で環境構築するなら ``Python 3.8.8, tensorflow==2.4.1`` になる。
    - pip install 指定したパッケージを明確にする。
        - 今回のコードでは [Readme.md](https://github.com/naltoma/trial-keras-bert) に書いてる通り ``pip install keras-bert tensorflow keras sentencepiece scikit-learn transformers matplotlib pydot`` としているため、これらが対象パッケージとなる。
- step a-3: keras/pytorchどちらか使うならベースとなるdockerhubを利用する。
    - [Nvidia公式のCUDAコンテナ](https://hub.docker.com/r/nvidia/cuda)も用意されているが、pythonすら入っていない最小限環境のため、これをベースに環境構築するのは手間がかかる。少なくとも少し試してみた限りでは難しそうな印象。
    - kerasを使うなら[tesorflow/tensorflow](https://hub.docker.com/r/tensorflow/tensorflow)をベースにしよう。なお、これはpipで環境構築されています。
    - pytorchを使うなら[pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch)をベースにしよう。なお、これはcondaで環境構築されています。（公式自体がconda推奨）
    - 今回はkerasを使うので、tensorflowコンテナをベースにすることに。これがstpe 4の1行目。
- step a-4: defファイルを作成する。
    - コンテナ作成方法はいくつか種類がある。sandboxを作って手動で作業したいなら[README.md](README.md)を参照しよう。今回は定義ファイル（defファイル）を作成して、それを使ってコンテナを作成することに。
    - [keras-bert.def](keras-bert.def)
        - 1行目: ``BootStrap: localimage``
            - 今回は pull で用意したコンテナイメージをベースに作成するため、``localimage`` と指定。もしdockerhubからダウンロードしたい場合には ``docker`` と書く。
        - 2行目: ``From: tensorflow_latest-gpu-py3.sif``
            - 今回は localimage なので、具体的なファイル名 ``tensorflow_latest-gpu-py3.sif`` を指定している。
            - dockerhubのtensorflowからダウンロードしたい場合には ``tensorflow/tensorflow:latest-gpu`` のように書く。書き方は dockerhub の pull で指定する記述と一緒。
        - 5行目: ``%post``
            - コンテナイメージを用意した後で何か追加処理をする場合に使う。
            - 今回は apt を最新に更新し、graphvizをインストール。その後で pip で関連パッケージをインストール。
              - ``apt-get update``
              - ``apt install -y --no-install-recommends graphviz``
              - ``pip install --upgrade pip``
              - ``pip install keras==2.3 keras-bert sentencepiece scikit-learn transformers matplotlib pydot``
            - これだけで良いのだけど、今回は ``keras==2.3`` として keras だけバージョンを指定している。バージョンを指定しないと自動で最新が選ばれてしまい、それが要求する tensorflow==2.4.1が自動でインストールされてしまう。
                - これは大きな問題であることに注意。GPU対応版として用意されていたのは ``tensorflow-gpu==2.1.0`` なのに、それとは別に tensorflow==2.4.1 がインストールされてしまう。そして keras はGPU非対応のtensorflowを参照してしまう。
                - この結果、見かけ上GPUに対応したパッケージも入っているが、実際にはそれを使わないという残念なコンテナを作ってしまう。残念なコンテナでプログラムを実行すると、nvida-smi で確認してもGPU上では動作していないことを確認できる。また、以下のようにして ``tf.test.gpu_device_name()`` の出力結果からも判断できる。
                - kerasバージョンを指定した場合の動作確認ログ: [logs/gpu-available.txt](logs/gpu-available.txt)
                    - 全部が ``I`` 出力されていて、GPU名 ``Tesla V100S-PCIE-32GB`` が確認できる。
                - kerasバージョンを指定しなかった場合の残念な動作確認ログ: [logs/gpu-unavailable.txt](logs/gpu-unavailable.txt)
                    - ところどころ ``W`` 出力されていて、GPUがskipされている。
- step a-5: 用意したdefファイルからコンテナを作成する。
    - ``singularity build --fakeroot コンテナ名.sif defファイル`` するだけ。
    - 途中でエラーが出る場合にはそこで終了してしまう。依存関係確認しながらとかinteractiveに操作したい場合には、以下のようにsandbox作成、shellモードで起動した状態で構築しよう。構築し終えたらbuildし直すこと。
        - sandbox作成: ``singularity build --fakeroot --sandbox test tensorflow_latest-gpu-py3.sif``
        - shellモードで起動: ``singularity shell --fakeroot --writable test``
        - sandboxからbuild: ``singularity build --fakeroot コンテナ名.sif test``

---
### (b) どうやってSlurmで動かすのか
- step b-1: バッチファイルを作成する。
    - [train.sbatch](train.sbatch)。ファイル名は何でも良いが、拡張子に ``.sbatch`` を付けておくと分かりやすい。
    - 1行目: ``#!/bin/bash``
        - インタプリタを指定するシバン（shebang）。
    - 2行目: ``#SBATCH --job-name keras-bert-normal``
        - Slurmでジョブ実行させる場合、自動でJOBIDが付与される。それだけだと何のジョブなのか分かりづらいた。名称をつけて判別しやすくするためのオプションがこれ。
    - 3行目: ``#SBATCH --output logs/%x-%j.log``
        - オプションの説明
            - ``%x`` は ``--job-name`` で付けたジョブ名。
            - ``%j`` はSlurmが自動付与するジョブID。
            - 今回の例では ``logs/keras-bert-normal-12345.log`` のようなファイル名を指定している。実行する都度上書きして良いなら ``#SBATCH --output result.txt`` のように書いてもOK。
        - Slurmでジョブ実行する場合、標準出力先が変更される。変更されたままでは面倒なので、出力先をファイルに指定すると良い。
        - **要注意**
            - 今回の例では ``logs/`` 以下のファイルを指定している。このように **ディレクトリを含む形で指定する場合には、必ずジョブ実行前に該当ディレクトリを手動で作成** すること。もし該当ディレクトリがないままジョブ実行すると、ターミナル上には何も出力されず、単に実行が終了してしまう。恐らくSlurm側のログには何かしら出力されているが、ユーザ側ターミナルには何も出力されない。
    - 4行目: ``#SBATCH --error logs/%x-%j.err``
        - 標準エラー出力先を指定している。
    - 5行目: ``#SBATCH --nodes 1``
        - 実行時に利用するノード数を指定。普通は1。複数ノードで通信しながら並列分散処理とかするプログラムならここで指定すると良いかもしれない。（未確認）
    - 6行目: ``#SBATCH --gpus tesla:1``
        - ノードに搭載されているGPU数の指定。変更できないのでこのまま。
    - 7行目以降
        - Slurmで実行したいコマンドラインの羅列。
        - ``singularity exec --nv ~/SIF-images/test.sif python train.py``
            - 用意したコンテナで ``python train.py`` を実行する際のコマンドライン例。
        - [MattermostのWebHook](https://ie.u-ryukyu.ac.jp/syskan/service/mattermost/)利用して、終了時にDM送るようにすることもできる（はず）。
- step b-2: ジョブを投入する。
    - ``sbatch バッチファイル名``
    - リソースが空き次第実行開始。
        - ジョブ管理状況の確認: ``squeue``
        - ジョブをキャンセルしたい場合: ``scancel JOBID``

---
### その他
- 一つのジョブは必要最小限で実行しよう。
    - 例えば深層学習のネットワーク構造を変えて実行したり、ニューロン数を変えて実行する等、複数のパターンを実行する状況を想定してみよう。このとき、実行中に途中で何らかの不都合でエラーになってしまった場合、本来なら終えていたはずのその前の処理を含めて全てやり直すことになる。これを避けるため、独立して実行可能なものは分けてジョブ実行した方が良い。4台あるので空いているなら4ジョブ並行して実行されるという意味でもお得。
    - つまり、一つのジョブでまとめてやるのではなく、最小限のジョブを多数投げる方が良い。
