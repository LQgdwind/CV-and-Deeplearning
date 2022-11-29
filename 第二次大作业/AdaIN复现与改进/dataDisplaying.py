from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
import os

class Multi_Animator:
    # 在一个画布中画多个折线图
    def __init__(self, fig_main, axes_main,xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), rows=1, cols=1,
                 figsize=(10, 7)):
        if legend is None:
            legend = []
        use_svg_display()
        self.label = xlabel
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
        # display.display(self.fig)


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

def plot_result_images(content, style, im_cs, fig,ax,epoch=0, directory="/root/trans/results"):
    ax[3][0].imshow(content)
    ax[3][0].axis('off')
    ax[4][0].imshow(style)
    ax[4][0].axis('off')
    ax[5][0].imshow(im_cs)
    ax[5][0].axis('off')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, f"results_{epoch}.png"))

def mu(x):
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    return x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

def sigma(x, eps=1e-5):
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    return std