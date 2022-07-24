import numpy as np
from typing import List
from typing import Callable

class Halpern(object):
    '''
        Halpernアルゴリズムのクラス.
    '''
    count = 0

    def __init__(self, alpha: float, a: float, T: Callable[[np.ndarray], np.ndarray],name: str=None) -> None:
        '''
            <引数>
                alpha:  0より大きく1未満のパラメータ.

                T:      不動点を求めたい非拡大写像.

                name:   このHalpernアルゴリズムの名前
        '''
        if not 0 < alpha < 1:
            raise ValueError(f'Invalid alpha value: {alpha}')
        if not 0 < a <= 1:
            raise ValueError(f'Invalid k value: {a}')
        self.alpha = alpha
        self.a = a
        self.T = T
        if name is None:
            self.name = f'HAL{Halpern.count}'
            Halpern.count += 1
        else:
            self.name = name
        self.history = None  # 結果の保存用辞書

    def __repr__(self) -> str:
        return self.history.__repr__()

    def solve(self, x0: np.ndarray, n_iter: int=10) -> dict:
        '''
            Halpernアルゴリズムを実行する。
            <引数>
                x0:     初期点を表す1次元のNumPy配列.

                n_iter: 反復回数. 初期値は10.
        '''
        
        alpha = self.alpha
        a = self.a
        T = self.T

        # 結果の保存用辞書
        history = {
                    'name': self.name,
                    'xk': [x0.tolist()],
                    'dist': [np.linalg.norm(x0 - T(x0))],
                    'final': None
                }
        
        xk = x0
        
        # Halpernアルゴリズムの本体.
        ## Halpernは x_k+1 = alpha_k * x_0 + (1 - alpha_k)T(x_k)
        ## alpha_k = 1 / k^a (k=0,1,2,3...としてよい？？？)
        k = 0
        for n in range(n_iter):
            if k==0:
                alpha_k = 0.999
            else:
                alpha_k = 1 / (k**a)
            k+=1
            xk = alpha_k * x0 + (1 - alpha_k) * T(xk)
            history['xk'].append(xk.tolist())  # 点の保存
            history['dist'].append(np.linalg.norm(xk - T(xk)))
        
        history['final'] = xk
        history['xk'] = np.array(history['xk'])
        self.history = history  

        return history
