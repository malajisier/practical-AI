**https://zhuanlan.zhihu.com/p/22252270**



## 1. SGD  

现在的SGD一般都指 `mini-batch gradient descent`。SGD就是每一次迭代计算mini-batch的梯度，然后对参数进行更新:  

$$
g_t=\nabla_{\theta_{t-1}}{f(\theta_{t-1})}
$$

$$
\Delta{\theta_t}=-\eta*g_t
$$

![\eta](https://www.zhihu.com/equation?tex=%5Ceta)是学习率，![g_t](https://www.zhihu.com/equation?tex=g_t)是梯度    

SGD完全依赖于当前batch的梯度，所以![\eta](https://www.zhihu.com/equation?tex=%5Ceta)可理解为允许当前batch的梯度多大程度影响参数更新   

**缺点**:   

- 选择初始学习率较为困难，太小收敛速度慢，太大会导致损失函数在最小值附近振荡，或偏离最小值
- 非凸函数存在大量的局部最优解或鞍点



## 2. Momentum

动量法：在当前梯度值上，加上上一次梯度值的衰减。

所以当前梯度方向与上一次方向相同的参数，加大更新力度；否则消减更新力度   

总而言之，较于SGD，动量法可以在相关方向加速收敛，抑制振荡
$$
m_t=\mu*m_{t-1}+g_t     
$$

$$
\Delta{\theta_t}=-\eta*m_t
$$



## 3. NAG      

Nesterov accelerated gradien   

nesterov项在梯度更新时做一个校正，避免前进太快，相当于预估了下一次参数所在的位置，提高了灵敏度  
$$
g_t=\nabla_{\theta_{t-1}}{f(\theta_{t-1}-\eta*\mu*m_{t-1})}
$$

$$
m_t=\mu*m_{t-1}+g_t
$$

$$
\Delta{\theta_t}=-\eta*m_t
$$





--------------------------------------------------------------------------------------------------------------------------------------------------

**以上均需要人工设置`lr`， 灵活性差，下面的算法均为 自适应学习率方法**

------------------------------------------------------------------------------------------------------------------------------



## 4. Adagrad      

Adagrad是一种自适应的梯度下降方法，它希望根据梯度的更新量调节梯度实际的影响值。如果一个变量的梯度更新量比较大，那么再后面的更新过程中我们会适当减少它的更新量，如果更新量比较小，那么我们可以适当地增加它的更新量。
$$
n_t=n_{t-1}+g_t^2
$$

$$
\Delta{\theta_t}=-\frac{\eta}{\sqrt{n_t+\epsilon}}*g_t
$$

此处，对![g_t](https://www.zhihu.com/equation?tex=g_t)从1到![t](https://www.zhihu.com/equation?tex=t)进行一个递推形成一个**约束项regularizer**，![-\frac{1}{\sqrt{\sum_{r=1}^t(g_r)^2+\epsilon}}](https://www.zhihu.com/equation?tex=-%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Csum_%7Br%3D1%7D%5Et%28g_r%29%5E2%2B%5Cepsilon%7D%7D)，![\epsilon](https://www.zhihu.com/equation?tex=%5Cepsilon)用来保证分母非0     

**特点：**  

- 数据稀疏时，收敛更快
- 前期![g_t](https://www.zhihu.com/equation?tex=g_t)较小的时候， regularizer较大，能够放大梯度
- 后期![g_t](https://www.zhihu.com/equation?tex=g_t)较大的时候，regularizer较小，能够约束梯度

**缺点：**

- 仍依赖于人工设置全局`lr`

- 长时间分母过大，梯度趋于0，无法有效更新参数         

  

## 5. Adadelta      

对Adagrad的改进，参照了牛顿法   
$$
E|g^2|_t=\rho*E|g^2|_{t-1}+(1-\rho)*g_t^2
$$

$$
\Delta{x_t}=-\frac{\sqrt{\sum_{r=1}^{t-1}\Delta{x_r}}}{\sqrt{E|g^2|_t+\epsilon}}
$$

其中，![E](https://www.zhihu.com/equation?tex=E)代表求期望。可以看出Adadelta已经不用依赖于全局学习率了

**特点：**

- 训练初中期，加速效果不错，很快

- 训练后期，反复在局部最小值附近抖动    

  

## 6. RMSprop     

Adagrad有一个很大的问题，那就是随着优化的进行，更新公式的分母项会变得越来越大。所以理论上更新量会越来越小。   

Rmsprop就试图解决这个问题。在它的算法中，分母的梯度平方和不再随优化而递增，而是做加权平均
$$
G_{t+1}=\beta*G_{t}+(1-\beta)*g^2
$$

$$
x_{t+1} = x_t - lr * \frac{g}{\sqrt{G_{t+1}} + \varepsilon }
$$

**特点：**

- 其实RMSprop依然依赖于全局学习率
- RMSprop算是Adagrad的一种发展，和Adadelta的变体，效果趋于二者之间
- 适合处理非平稳目标 - 对于RNN效果很好



## 7. Adam

Adaptive Moment Estimation = RMSprop + Momentum   

Adam本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳
$$
m_t=\mu*m_{t-1}+(1-\mu)*g_t
$$

$$
n_t=\nu*n_{t-1}+(1-\nu)*g_t^2
$$

$$
\hat{m_t}=\frac{m_t}{1-\mu^t}
$$

$$
\hat{n_t}=\frac{n_t}{1-\nu^t}
$$
$$
\Delta{\theta_t}=-\frac{\hat{m_t}}{\sqrt{\hat{n_t}}+\epsilon}*\eta
$$



![m_t](https://www.zhihu.com/equation?tex=m_t)，![n_t](https://www.zhihu.com/equation?tex=n_t)分别是对梯度的一阶矩估计和二阶矩估计，可以看作对期望![E|g_t|](https://www.zhihu.com/equation?tex=E%7Cg_t%7C)，![E|g_t^2|](https://www.zhihu.com/equation?tex=E%7Cg_t%5E2%7C)的估计    

![\hat{m_t}](https://www.zhihu.com/equation?tex=%5Chat%7Bm_t%7D)，![\hat{n_t}](https://www.zhihu.com/equation?tex=%5Chat%7Bn_t%7D)是对![m_t](https://www.zhihu.com/equation?tex=m_t)，![n_t](https://www.zhihu.com/equation?tex=n_t)的校正，这样可以近似为对期望的无偏估计     

可以看出，直接对梯度的矩估计对内存没有额外的要求，而且可以根据梯度进行动态调整，而![-\frac{\hat{m_t}}{\sqrt{\hat{n_t}}+\epsilon}](https://www.zhihu.com/equation?tex=-%5Cfrac%7B%5Chat%7Bm_t%7D%7D%7B%5Csqrt%7B%5Chat%7Bn_t%7D%7D%2B%5Cepsilon%7D)对学习率形成一个动态约束，而且有明确的范围。

**特点：**

- 结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点
- 对内存需求较小
- 为不同的参数计算不同的自适应学习率
- 也适用于大多非凸优化 - 适用于大数据集和高维空间

$$

$$
## 8. Adamax

是Adam的一种变体，此方法对学习率的上限提供了一个更简单的范围
$$
n_t=max(\nu*n_{t-1},|g_t|)
$$

$$
\Delta{x}=-\frac{\hat{m_t}}{n_t+\epsilon}*\eta
$$
Adamax学习率的边界范围更简单



## 9. Nadam    

Nadam类似于带有Nesterov动量项的Adam
$$
\hat{g_t}=\frac{g_t}{1-\Pi_{i=1}^t\mu_i}
\$$
$$
m_t=\mu_t*m_{t-1}+(1-\mu_t)*g_t
$$

$$
\hat{m_t}=\frac{m_t}{1-\Pi_{i=1}^{t+1}\mu_i}
$$

$$
n_t=\nu*n_{t-1}+(1-\nu)*g_t^2
$$

$$
\hat{n_t}=\frac{n_t}{1-\nu^t}\bar{m_t}=(1-\mu_t)*\hat{g_t}+\mu_{t+1}*\hat{m_t}
$$

$$
\Delta{\theta_t}=-\eta*\frac{\bar{m_t}}{\sqrt{\hat{n_t}}+\epsilon}
$$


可以看出，Nadam对学习率有了更强的约束，同时对梯度的更新也有更直接的影响。一般而言，在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果。



## 经验之谈

- 对于稀疏数据，尽量使用学习率可自适应的优化方法，不用手动调节，而且最好采用默认值
- SGD通常训练时间更长，但是在好的初始化和学习率调度方案的情况下，结果更可靠
- 如果在意更快的收敛，并且需要训练较深较复杂的网络时，推荐使用学习率自适应的优化方法。
- Adadelta，RMSprop，Adam是比较相近的算法，在相似的情况下表现差不多。
- 在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果