vLLM中通过继承`CustomOp`来定义`Op`。针对不同后端，生成不同的forward实现。

## 代码参考：
- `CustomOp`定义：https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/custom_op.py

- `register`函数解析：
    在自定义Op时候，会在`CustomOp`的`op_registry`字典中注册`op`，用于控制自定义`op`的行为。
    ```python
    @classmethod
    def register(cls, name: str):

        def decorator(op_cls):
            assert name not in cls.op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            cls.op_registry[name] = op_cls
            return op_cls

        return decorator

    @register("layer_norm_npu")
    class LayerNormNpu(CustomOp):
        ...
    ```
- `forward` - `dispatch`逻辑解析：  
    具体的行为取决于是否启用custom op的实现：  
    - 如果关闭，则走`forward_native`实现，一般是通过torch op拼接完成特定功能；
    - 如果开启，则通过`dispatch`方法走特定后端的实现。    

    判断是否启用`custom op`的方法：  
    - 若继承`CustomOp`时没有注册，则通过`CustomOp`的全局控制方法进行控制，即`vLLM`的`compilation_config`的控制层级小于piecewise，且`custom_ops`不为None，即全局关闭所有custom op。
    - 若继承`CustomOp`时进行注册，则判断`vLLM`的`compilation_config`中是否开启了全局编译，或者指定编译。具体为`custom_ops`域为`all`或者包含`+{name}`。

    ```python
        def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cpu(self, *args, **kwargs):
        ...

    def forward_oot(self, *args, **kwargs):
        # By default, we assume that OOT ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        compilation_config = get_current_vllm_config().compilation_config
        enabled = self.enabled()
        if enabled:
            compilation_config.enabled_custom_ops.update([self.__class__.name])
        else:
            compilation_config.disabled_custom_ops.update(
                [self.__class__.name])

        if not enabled:
            return self.forward_native

        if current_platform.is_rocm():
            return self.forward_hip
        elif current_platform.is_cpu():
            return self.forward_cpu
        elif current_platform.is_out_of_tree():
            return self.forward_oot
        else:
            return self.forward_cuda
    ```