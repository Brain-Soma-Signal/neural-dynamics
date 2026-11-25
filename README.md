# Neural Dynamics

Soma & Signal 的神经算子库。研究普通投影层如何在保留原有权重的同时，把一次输入形成的状态带入下一次计算。

- [FusedLinear](src/neural_dynamics/fused_unit.py) 将状态读取、暂存与提交分开。
- [树突分支](src/neural_dynamics/dendritic_cell.py) 在输出单元内部计算多个非线性分支。
- [层迁移与控制器](src/neural_dynamics/dendritic_qwen.py) 将这些算子接入模型，统一管理各层的状态。

## 本地验证

在独立 Python 环境中，从仓库根目录运行：

```sh
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
```

首版源自 AI-New_Architecture 第 46、47、53 阶段。包名为 `neural_dynamics`。组件测试使用微型张量；完整模型的实验与独立重放审计由原研究工作区保管。

[Soma & Signal 组织首页](https://github.com/Brain-Soma-Signal)
