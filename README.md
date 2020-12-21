# deeplearning-framework
动手实现一个深度学习框架:HuichuanFlow


通过把机器学习的理论、原理和工程实践结合起来，在动手“建造”中加深对理论的“理解”。深度学习框架正是一个最好的例子。但是，以TensorFlow为代表的现代深度学习框架的源码对于学习而言已经过于复杂，因此，我们决定用纯Python实现一个简单的深度学习框架HuichuanFlow（核心代码2K行左右，只依赖了Numpy）。这个框架基于计算图，支持自动求导和梯度下降优化算法（及变体）。我们用该框架搭建了一些经典模型，包括LR、FM、DNN、CNN、RNN、W&D、DeepFM等）。该框架还包括一些工程上的方案，包括训练器、分布式训练、模型部署和服务等。


我在代码中写了尽量详细的注释，通过阅读源码亦能理解这个“麻雀虽小五脏俱全”的框架，并从中学习和理解机器学习背后的原理。

著名物理学家，诺贝尔奖得主[Richard Feynman](https://en.wikipedia.org/wiki/Richard_Feynman)办公室的黑板上写了："What I cannot create, I do not understand."。

### 特性

- 基于计算图，可用于搭建大多数的机器学习模型。
- 支持自动求导。
- 支持随机梯度下降优化算法及其几个重要变种（如RMSProp、ADAGrad、ADAM等）。



