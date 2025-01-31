# ComfyUI-ColorshiftColor

## これは何？
これはComfyUIのカスタムノードです。
- 指定数で減色して色相/彩度/輝度の変更を行う
- 各パラメーターのランダム指定可能
- 色番号でマスク（色を変えない）のオンオフ可能
- マスクの反転がオンオフ可能

以上の機能を実装したシンプルなノードです。  

![image](https://github.com/852wa/ComfyUI-ColorshiftColor/blob/master/example/workflow%20(2).png)


- lock_color_num
  パレットのプレビューで割り振られた番号を,カンマで入力することでマスクに指定できます。


scikit-learnが必要なのでpip install scikit-learnを環境に入れてください。


![image](https://github.com/852wa/ComfyUI-ColorshiftColor/blob/master/example/samplecsc.gif)

連番対応

文字入力スペースの
##{"index": 0, "color": [0.0, 0.0, 1.0]}の##を外すと文字での色指定が出来ます。（複数可能）

## update
25.02.01 - mask機能の修正、CsCノードの高速化、exampleデータの見直し
