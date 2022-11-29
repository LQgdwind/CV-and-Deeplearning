from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

class Accumulator:
    """累加器"""
    def __init__(self, n):
        self.data = [0.0] * n
        # data为1*n的列表
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        # 将存储的data与传进来的参数依次相加。
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Multi_Animator:
    # 在一个画布中画多个折线图
    def __init__(self, fig_main, axes_main,xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), rows=1, cols=1,
                 figsize=(4, 3)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig = fig_main
        self.axes = axes_main[rows][cols]
        self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def use_svg_display():
    # 在colab上规范图像格式
    backend_inline.set_matplotlib_formats('svg')

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    # 为图像设定信息

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
