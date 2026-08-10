import torch, pennylane as qml, time
print("torch:", torch.__version__, "| pennylane:", qml.version(), "| numpy:", __import__("numpy").__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("weights_only:", "weights_only" in torch.load.__doc__)

dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n, L = 4, 2

for method in ["backprop", "adjoint"]:
    dev = qml.device("default.qubit", wires=n)
    @qml.qnode(dev, interface="torch", diff_method=method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n))
        return [qml.expval(qml.PauliX(i)) for i in range(n)]

    torch.manual_seed(0)
    layer = qml.qnn.TorchLayer(circuit, {"weights": (L, n, 3)},
                init_method={"weights": lambda w: torch.nn.init.normal_(w, 0.0, 0.1)}).to(dev_t)
    torch.manual_seed(0)
    z = torch.randn(8, n, device=dev_t, requires_grad=True)
    out = layer(torch.tanh(z) * (torch.pi/2))
    out.sum().backward()
    print(f"{method:10s} out.sum={out.sum().item():.10f}  z.grad={z.grad.abs().sum().item():.10f}  "
          f"w.grad={layer.weights.grad.abs().sum().item():.10f}")

# GPU timing at the worst case
if torch.cuda.is_available():
    n16 = 16
    dev = qml.device("default.qubit", wires=n16)
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def c16(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n16), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n16))
        return [qml.expval(qml.PauliX(i)) for i in range(n16)]
    lay = qml.qnn.TorchLayer(c16, {"weights": (4, n16, 3)}).to(dev_t)
    zz = torch.randn(32, n16, device=dev_t, requires_grad=True)
    lay(torch.tanh(zz)*(torch.pi/2)).sum().backward()   # warmup
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    lay(torch.tanh(zz)*(torch.pi/2)).sum().backward()
    torch.cuda.synchronize()
    print(f"d=16 L=4: {time.perf_counter()-t0:.3f}s/step, {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# The bug this upgrade exists to fix
torch.save({"a": torch.randn(3)}, "/tmp/_t.pt")
print("weights_only load:", torch.load("/tmp/_t.pt", weights_only=True))
