For an arbitary circuit $U$ preparing a 4-qubits GHZ state, the **unitary** matrix of $U$ should satisfy the following structure:

$$
\begin{array}{ll}
U \left| 0000 \right> = \frac{1}{\sqrt 2} (\left| 0000 \right> + \left| 1111 \right>) \\
\begin{bmatrix}
  \frac{1}{\sqrt 2} & 0 & \cdots & 0 & \frac{1}{\sqrt 2} \\
  0 & & & &  \\
  \vdots & & \large{*} & & \\
  0 &  & & & \\
  \frac{1}{\sqrt 2} & & & & \\
\end{bmatrix} \begin{bmatrix}
  1 \\
  0 \\
  \vdots \\
  0 \\
  0 \\
\end{bmatrix} = \begin{bmatrix}
  \frac{1}{\sqrt 2} \\
  0 \\
  \vdots \\
  0 \\
  \frac{1}{\sqrt 2} \\
\end{bmatrix}
\end{array}
$$

where $*$ represents any number.

Hence $ U \left| 0110 \right> $ just projects the 6-th column of $U$, which could result **anything** except $\left| 0000 \right>$, since the first element is 0!

ℹ 这，就是正交性 :(
