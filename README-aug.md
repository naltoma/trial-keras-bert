# train_aug.py (augmentation付きスクリプトの補足)
## データ水増しの仕方
[SentencePiece](https://github.com/google/sentencepiece)のSubword regularizationを使って水増ししてみる。SentencePieace.encode()にて `` enable_sampling=True, nbest_size=-1, alpha=0.01`` ぐらいのオプションを付けるとある程度のランダムサンプリングをしてくれる。以下、サンプリング例。

```python
>>> import os
>>> pretrained_path = os.path.expanduser('~/model/bert-wiki-ja/')
>>> sentencepiece_path = os.path.join(pretrained_path, 'wiki-ja.model')
>>> 
>>> import sentencepiece as spm
>>> sp = spm.SentencePieceProcessor()
>>> sp.Load(sentencepiece_path)
True
>>> 
>>> sentence = "これはテストです。"
>>> 
>>> # 10回ランダム生成
>>> for i in range(10):
...     print(sp.encode(sentence, out_type=str, enable_sampling=True, nbest_size=-1, alpha=0.01))
... 
['▁', 'これは', 'テ', 'スト', 'です', '。']
['▁これは', 'テスト', 'で', 'す', '。']
['▁', 'こ', 'れ', 'は', 'テス', 'ト', 'で', 'す', '。']
['▁', 'これ', 'は', 'テス', 'ト', 'で', 'す', '。']
['▁', 'こ', 'れ', 'は', 'テス', 'ト', 'で', 'す', '。']
['▁これ', 'は', 'テスト', 'です', '。']
['▁', 'こ', 'れ', 'は', 'テ', 'スト', 'で', 'す', '。']
['▁これは', 'テス', 'ト', 'で', 'す', '。']
['▁', 'これ', 'は', 'テ', 'スト', 'で', 'す', '。']
['▁これ', 'は', 'テス', 'ト', 'です', '。']
>>> # 最適サブワード生成だと毎回同じ最適1件を出力。
>>> print(sp.encode(sentence, out_type=str))
['▁これは', 'テスト', 'です', '。']
>>> print(sp.encode(sentence, out_type=str))
['▁これは', 'テスト', 'です', '。']
```

<hr>

## 実装方針
- チートを避けるため、全データを train, test, val に分けたあと、trainだけを水増しする。水増し割合は実装の都合で整数倍に限定。実装上は train をN倍（同じデータセットを後ろにN回append）した上で SentencePiece のサブワード化してるので大規模データではそれだけメモリを食います。
- 関連パラメータ
    - ``aug_size``: 水増しサイズ。1なら水増しなし。2なら2倍。
    - ``test_rate``: データセットにおけるテストデータセットの割合。
    - ``val_rate``: テストを除外した残りにおけるバリデーションの割合。
