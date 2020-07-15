# パラメーターを別ファイルで管理したいので分ける。
# ハイパーパラメーターの管理はどうするのが良いのか......
# 他の人(特に強い人)を参考に考える。クラスにするほどでも無い気がする。
class HyperParameters:  # 名前、分布、下限、上限、(配列)を決めればOptunaに渡せる。
    raise NotImplementedError


class OptunaTrialParameters:
    raise NotImplementedError


class FitParameters:
    raise NotImplementedError