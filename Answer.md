# Answer

    二切虚张声势的狐假虎威。

----

### Single Choices

#### Q1: logical clause to pauli operator

ℹ 同去年第九题，已加入题库原题 🎉

$$
\begin{array}{ll}
x_1 \lor x_2 
  &= x_1 + x_2 - x_1 \land x_2 \left( x = \frac{I - Z}{2} \right) \\
  &= \frac{1}{2}(I - Z_1) + \frac{1}{2} (I - Z_2) - \frac{1}{4} (I - Z_1) * (I - Z_2) \\
  &= \frac{3}{4} I - \frac{1}{4} (Z_1 + Z_2 + Z_1*Z_2) \\
\end{array}
$$

#### Q2: representing H and S gate in Rodrigues' form

定义 $ R_{\widetilde{n}}(\theta) = e^{-iθ / 2 \hat{n} \vec\sigma} = \mathrm{cos}(\theta/2) I - i \mathrm{sin}(n_x X + n_y Y + n_z Z) $，散见于:

  - [mindquantum.core.gates.Rn](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.Rn.html)
  - [Markus Schmassmann's lecture notes](https://qudev.phys.ethz.ch/static/content/courses/QSIT07/presentations/Schmassmann.pdf)
  - [Wikipedia rotation gates](https://en.wikipedia.org/wiki/List_of_quantum_logic_gates#Rotation_operator_gates)

#### Q3: finding phase of expm(A)

ℹ 题库里有同模板的题 :)

根据[定理](https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%98%B5%E6%8C%87%E6%95%B0): 若 $A = U D U^{-1}$ 其中 $D$ 为对角阵，则 $ e^A = U e^D U^{-1}$

$$
e^A = \frac{1}{\sqrt 5} \begin{bmatrix}
  1 & 2 \\
  -2 & 1 \\
\end{bmatrix} = U \frac{1}{\sqrt 5} \begin{bmatrix}
  1 + 2i & 0 \\
  0 & 1 - 2i \\
\end{bmatrix} U^{-1} = U e^D U^{-1} \\
A = U log(e^D) U^{-1} = -i \theta Y \\
$$

故有 $ \theta I = U log(e^D) U^{-1} (-iY)^{-1} = -arccos(\frac{1}{\sqrt 5}) I $，即解出 $ \theta = -arccos(\frac{1}{\sqrt 5}) $

#### Q4: quantum projective measure

ℹ 题库里有原题 🎉

$$
\begin{array}{ll}
\left| M_1 | \psi \right> 
= \frac{1}{3} \begin{bmatrix}
  1 & -i & 0 & 1 \\
  i & 1 & 0 & i \\
  0 & 0 & 0 & 0 \\
  1 & -i & 0 & 1 \\
\end{bmatrix} \begin{bmatrix}
  \frac{\sqrt 2}{2} \\
  \frac{1}{2} \\
  0 \\
  -\frac{1}{2} \\
\end{bmatrix} = \frac{1}{6} \begin{bmatrix}
  (\sqrt 2 - 1) - i \\
  1 + (\sqrt 2 - 1) i \\
  0 \\
  (\sqrt 2 - 1) - i \\
\end{bmatrix} \\
Prob(M_1) = \left | \left< \psi | M_1^\dagger M_1 | \psi \right> \right| = \frac{2 - \sqrt 2}{6}
\end{array}
$$

#### Q5: gate composition

$$
\begin{array}{ll}
H * T * H * T 
&= \frac{1}{\sqrt 2} \begin{bmatrix}
  1 & 1 \\
  1 & -1 \\
\end{bmatrix} \begin{bmatrix}
  1 & 0 \\
  0 & e^{i\pi/4} \\
\end{bmatrix} \frac{1}{\sqrt 2} \begin{bmatrix}
  1 & 1 \\
  1 & -1 \\
\end{bmatrix} \begin{bmatrix}
  1 & 0 \\
  0 & e^{i\pi/4} \\
\end{bmatrix} \\ 
&= \frac{1}{2} \begin{bmatrix}
  1 + e^{i\pi/4} & e^{i\pi/4} (1 - e^{i\pi/4}) \\
  1 - e^{i\pi/4} & e^{i\pi/4} (1 + e^{i\pi/4}) \\
\end{bmatrix} \\
&= \frac{1}{2} \begin{bmatrix}
  1 + e^{i\pi/4} & e^{i\pi/4} - i \\
  1 - e^{i\pi/4} & e^{i\pi/4} + i \\
\end{bmatrix} 
\end{array}
$$

### Multiuple Choices

#### Q1: judging statements

A选项错误，考虑对|0>而言共计绕z轴旋转90°、绕x轴旋转45°，显然不可能实现H门的均匀打散效果  
B选项错误，考虑矩阵形式，Rz和H的组合只含指数函数，而Rx含三角函数
C选项正确，否则CNOt无法产生比特纠缠  
D选项正确，易验证

$$
\begin{array}{ll}
Y Z Y^\dagger
= \begin{bmatrix}
  0 & -i \\
  i & 0 \\
\end{bmatrix} \begin{bmatrix}
  1 & 0 \\
  0 & -1 \\
\end{bmatrix} \begin{bmatrix}
  0 & -i \\
  i & 0 \\
\end{bmatrix}
= \begin{bmatrix}
  -1 & 0 \\
  0 & 1 \\
\end{bmatrix} = -Z
\end{array}
$$

#### Q2: gate composition
#### Q3: state equivalence

模拟可得

#### Q4: quantum circuit

模拟可得，注意GHZ线路的实现，要用控制位倒置版本的 CNOT 门

$$
GHZ_4 = \mathrm{H}(0) \rightarrow \mathrm{CNOT}^{ctrl=0}(1) \rightarrow \mathrm{CNOT}^{ctrl=1}(2) \rightarrow \mathrm{CNOT}^{ctrl=2}(3)
$$

#### Q5: gate composition

本质为 2-qubit QFT 线路分解，参见 [https://zhuanlan.zhihu.com/p/474941485](https://zhuanlan.zhihu.com/p/474941485)  

$$
QFT_2 = \mathrm{H}(0) \rightarrow \mathrm{S}^{ctrl=0}(1) \rightarrow \mathrm{H}(1) \rightarrow \mathrm{SAWP}(0, 1)
$$

A选项正确；B选项受控旋转门构造错误；C选项X门不对称；D选项受控旋转门控制比特反了

### Programming problem

简单 VQA 拟合指定概率分布的问题，见 [solutions/P1.py](solutions/P1.py)

----
by Armit
2024/04/15
