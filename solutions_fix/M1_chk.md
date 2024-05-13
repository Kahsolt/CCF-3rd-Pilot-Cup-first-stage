验证B选项: `H·Rz(θ)·H = Rx(θ)`

依定义

$$
\begin{array}{ll}
H &= \frac{1}{\sqrt 2} \begin{bmatrix}
  1 & 1 \\
  1 & -1 \\
\end{bmatrix},
Rz(\theta) &= \begin{bmatrix}
  e^{-i\theta/2} & 0 \\
  0 & e^{i\theta/2} \\
\end{bmatrix},
Rx(\theta) &= \begin{bmatrix}
  cos(\theta/2) & -i sin(\theta/2) \\
  -i sin(\theta/2) & cos(\theta/2) \\
\end{bmatrix}
\end{array}
$$

即有 

$$
\begin{array}{ll}
H Rz(\theta) H &= \frac{1}{\sqrt 2} \begin{bmatrix}
  1 & 1 \\
  1 & -1 \\
\end{bmatrix} \begin{bmatrix}
  e^{-i\theta/2} & 0 \\
  0 & e^{i\theta/2} \\
\end{bmatrix} \frac{1}{\sqrt 2} \begin{bmatrix}
  1 & 1 \\
  1 & -1 \\
\end{bmatrix} \\
&= \frac{1}{2} \begin{bmatrix}
  1 & 1 \\
  1 & -1 \\
\end{bmatrix} \begin{bmatrix}
  e^{-i\theta/2} & e^{-i\theta/2} \\
  e^{i\theta/2} & -e^{i\theta/2} \\
\end{bmatrix} \\
&= \frac{1}{2} \begin{bmatrix}
  e^{-i\theta/2}+e^{i\theta/2} & e^{-i\theta/2}-e^{i\theta/2} \\
  e^{-i\theta/2}-e^{i\theta/2} & e^{-i\theta/2}+e^{i\theta/2} \\
\end{bmatrix} \\
&= \frac{1}{2} \begin{bmatrix}
  C & S \\
  S & C \\
\end{bmatrix} \\
\end{array}
$$

$$
\begin{array}{ll}
C &= e^{-i\theta/2}+e^{i\theta/2} \\
&= cos(-\theta/2) + i sin(-\theta/2) + cos(\theta/2) + i sin(\theta/2) \\
&= 2 cos(\theta/2)
\end{array} \\
\begin{array}{ll}
S &= e^{-i\theta/2}-e^{i\theta/2} \\
&= cos(-\theta/2) + i sin(-\theta/2) - cos(\theta/2) - i sin(\theta/2) \\
&= -2i sin(\theta/2)
\end{array}
$$

$$
\therefore H Rz(\theta) H = Rx(\theta)
$$
