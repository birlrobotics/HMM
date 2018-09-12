from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


pos_plot = None
ori_x = None
ori_y = None
ori_z = None
ori_w = None
fx = None
fy = None
fz = None
mx = None
my = None
mz = None


def init_plots():
    global pos_plot
    global ori_x
    global ori_y
    global ori_z
    global ori_w
    global fx
    global fy
    global fz
    global mx
    global my
    global mz
    fig = plt.figure()
    pos_plot = fig.add_subplot(111, projection='3d')
    fig = plt.figure()
    ori_xyz = fig.add_subplot(211, projection='3d')

    ori_x = fig.add_subplot(411)
    ori_y = fig.add_subplot(412)
    ori_z = fig.add_subplot(413)
    ori_w = fig.add_subplot(414)

    fig = plt.figure()
    fx = fig.add_subplot(231)
    fy = fig.add_subplot(232)
    fz = fig.add_subplot(233)
    mx = fig.add_subplot(234)
    my = fig.add_subplot(235)
    mz = fig.add_subplot(236)

def plot_legend():
    global pos_plot
    global ori_x
    global ori_y
    global ori_z
    global ori_w
    global fx
    global fy
    global fz
    global mx
    global my
    global mz

    pos_plot.legend()
    ori_x.legend()
    ori_y.legend()
    ori_z.legend()
    ori_w.legend()
    fx.legend()
    fy.legend()
    fz.legend()
    mx.legend()
    my.legend()
    mz.legend()


def plot_one_df(df, color, label):
    global pos_plot
    global ori_x
    global ori_y
    global ori_z
    global ori_w
    global fx
    global fy
    global fz
    global mx
    global my
    global mz

    if '.endpoint_state.pose.position.x' in df:
        pos_plot.plot(
            df['.endpoint_state.pose.position.x'].tolist(), 
            df['.endpoint_state.pose.position.y'].tolist(), 
            df['.endpoint_state.pose.position.z'].tolist(), 
            color=color,
            label=label
        )
        pos_plot.set_title("pos xyz")

        ori_x.plot(
            df.index.tolist(),
            df['.endpoint_state.pose.orientation.x'].tolist(), 
            color=color,
            label=label
        )
        ori_x.set_title("ori x")

        ori_y.plot(
            df.index.tolist(),
            df['.endpoint_state.pose.orientation.y'].tolist(), 
            color=color,
            label=label
        )
        ori_y.set_title("ori y")

        ori_z.plot(
            df.index.tolist(),
            df['.endpoint_state.pose.orientation.z'].tolist(), 
            color=color,
            label=label
        )
        ori_z.set_title("ori z")

        ori_w.plot(
            df.index.tolist(),
            df['.endpoint_state.pose.orientation.w'].tolist(), 
            color=color,
            label=label
        )
        ori_w.set_title("ori w")


    if '.wrench_stamped.wrench.force.x' in df:
        fx.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.force.x'].tolist(), 
            color=color,
            label=label)
        fx.set_title("fx")

        fy.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.force.y'].tolist(), 
            color=color,
            label=label)
        fy.set_title("fy")

        fz.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.force.z'].tolist(), 
            color=color,
            label=label)
        fz.set_title("fz")

        mx.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.torque.x'].tolist(), 
            color=color,
            label=label)
        mx.set_title("mx")

        my.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.torque.y'].tolist(), 
            color=color,
            label=label)
        my.set_title("my")

        mz.plot(
            df.index.tolist(),
            df['.wrench_stamped.wrench.torque.z'].tolist(), 
            color=color,
            label=label)
        mz.set_title("mz")

def show_plots():
    plt.tight_layout()
    plt.show()
