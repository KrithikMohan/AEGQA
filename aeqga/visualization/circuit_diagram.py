"""Draw paper-style circuits for the AEQGA population subsets.

Matches the rotation/CNOT notation of Sarracino et al., Figure 2, while
showing this project's 32-individual setup and crossover-before-mutation
sequence. The selected qubits are one illustrative random realization.
Run with the project virtual environment to export PNG, SVG, and PDF.
"""
from pathlib import Path
from aeqga.paths import output_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle
import numpy as np
from qiskit import QuantumCircuit


INK = "#202329"
GREEN = "#22af4c"
ROTATION = "#aa1552"
CNOT = "#073ab5"
CLASSICAL = "#7c8c9e"


def example_circuit(n_qubits):
    """Use an exact RY/CX decomposition of the code's two CRY(pi/2) gates."""
    circuit = QuantumCircuit(n_qubits, n_qubits)
    values = np.arange(1, 2**n_qubits + 1, dtype=float)
    circuit.initialize(values / np.linalg.norm(values), range(n_qubits))
    # CRY(theta) = RY(theta/2), CX, RY(-theta/2), CX on the target.
    for control, target in [(0, 1), (1, 0)]:
        circuit.ry(np.pi / 4, target)
        circuit.cx(control, target)
        circuit.ry(-np.pi / 4, target)
        circuit.cx(control, target)
    circuit.rx(np.pi / 2, 2)
    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def gate(ax, x, y, label, angle):
    ax.add_patch(Rectangle((x - .34, y - .30), .68, .60,
                           facecolor=ROTATION, edgecolor=ROTATION, zorder=3))
    ax.text(x, y + .08, label, ha="center", va="center",
            color="white", fontsize=12, zorder=4)
    ax.text(x, y - .16, angle, ha="center", va="center",
            color="white", fontsize=9, zorder=4)


def draw_circuit(ax, circuit, title):
    n = circuit.num_qubits
    ys = [-i for i in range(n)]
    bus = -n - .08
    ax.set_xlim(-.8, 17.2)
    ax.set_ylim(bus - .5, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=16)

    # Full-length wire paths, without resized images or clipped circuit panels.
    for i, y in enumerate(ys):
        ax.plot([.3, 16.9], [y, y], color=INK, lw=1.35, zorder=1)
        ax.text(-.10, y, rf"$q_{i}$", ha="right", va="center", fontsize=13)
        ax.add_patch(Rectangle((.48, y-.27), .60, .54,
                               facecolor="#e4e4e4", edgecolor="none", zorder=3))
        ax.text(.78, y, r"$|0\rangle$", ha="center", va="center", fontsize=11, zorder=4)
    for offset in [-.035, .035]:
        ax.plot([.3,16.9], [bus+offset,bus+offset], color=CLASSICAL, lw=1.2)
    ax.text(-.10,bus,r"$c$",ha="right",va="center",fontsize=13)
    ax.plot([.50,.64],[bus-.12,bus+.12],color=CLASSICAL,lw=1.2)
    ax.text(.57,bus+.17,str(n),fontsize=9,ha="center",color=CLASSICAL)

    # State preparation has the same full-register representation as Fig. 2.
    ax.add_patch(Rectangle((1.55, ys[-1]-.33), 2.25, n-1+.66,
                           facecolor=GREEN, edgecolor="none", zorder=3))
    for i,y in enumerate(ys):
        ax.text(1.69,y,str(i),color="white",fontsize=10,va="center",zorder=4)
    ax.text(2.70,ys[-1]/2,"State preparation",ha="center",va="center",
            color="white",fontsize=9,zorder=4)
    ax.text(2.70,ys[-1]/2-.32,r"$\mathrm{initialize}(\hat{\mathbf{x}})$",
            ha="center",va="center",color="white",fontsize=8,zorder=4)

    for x in [4.2,13.5]:
        ax.plot([x,x],[ys[-1]-.4,.4],color="#b7b7b7",lw=9,alpha=.35)
        ax.plot([x,x],[ys[-1]-.4,.4],color="#717171",lw=.9,ls="--")
    ax.text(2.67,.83,"Amplitude encoding",ha="center",fontsize=10,color="#555555")
    ax.text(8.35,.83,"Crossover",ha="center",fontsize=10,color="#555555")
    ax.text(12.6,.83,"Mutation",ha="center",fontsize=10,color="#555555")
    ax.text(15.4,.83,"Readout",ha="center",fontsize=10,color="#555555")

    operations = [instruction for instruction in circuit.data
                  if instruction.operation.name in {"ry","cx","rx"}]
    for column,instruction in enumerate(operations):
        x = 4.95 + column*.95
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        name = instruction.operation.name
        if name == "cx":
            control,target = [ys[q] for q in qubits]
            ax.plot([x,x],[control,target],color=CNOT,lw=1.6,zorder=2)
            ax.add_patch(Circle((x,control),.085,color=CNOT,zorder=4))
            ax.add_patch(Circle((x,target),.21,color=CNOT,zorder=4))
            ax.plot([x-.14,x+.14],[target,target],color="white",lw=1.5,zorder=5)
            ax.plot([x,x],[target-.14,target+.14],color="white",lw=1.5,zorder=5)
        else:
            angle = float(instruction.operation.params[0])
            label = r"$R_y$" if name == "ry" else r"$R_x$"
            angle_text = (r"$\pi/2$" if name == "rx" else
                          r"$\pi/4$" if angle>0 else r"$-\pi/4$")
            gate(ax,x,ys[qubits[0]],label,angle_text)

    for i,y in enumerate(ys):
        x = 14.15+i*.65
        ax.add_patch(Rectangle((x-.25,y-.27),.5,.54,
                               facecolor="#dfdfdf",edgecolor="none",zorder=3))
        ax.add_patch(Arc((x,y-.07),.34,.31,theta1=0,theta2=180,
                         color=INK,lw=1.2,zorder=4))
        ax.annotate("",xy=(x+.12,y+.17),xytext=(x-.025,y-.08),
                    arrowprops=dict(arrowstyle="-",color=INK,lw=1.2),zorder=4)
        for offset in [-.023,.023]:
            ax.plot([x+offset,x+offset],[y-.28,bus+.17],color=CLASSICAL,lw=.9,zorder=2)
        ax.annotate("",xy=(x,bus+.015),xytext=(x,bus+.20),
                    arrowprops=dict(arrowstyle="-|>",color=CLASSICAL,lw=1.0))
        ax.text(x+.12,bus+.14,str(i),fontsize=9,color=CLASSICAL)


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none",
                         "pdf.fonttype":42})
    fig,axes = plt.subplots(2,1,figsize=(14,9),
                            gridspec_kw={"height_ratios":[5.7,4.7]})
    fig.subplots_adjust(left=.055,right=.98,bottom=.12,top=.89,hspace=.38)
    fig.suptitle("AEQGA quantum circuits",fontsize=18,fontweight="bold",y=.98)
    fig.text(.5,.937,r"$n_p=32$  ·  Separate circuits for $H_0$ and $\Omega_M$",
             ha="center",fontsize=11,color="#555555")
    draw_circuit(axes[0],example_circuit(4),
                 "(a) Random subset  ·  16 individuals / 4 qubits")
    draw_circuit(axes[1],example_circuit(3),
                 "(b) Elite-copy subset  ·  8 individuals / 3 qubits")
    fig.text(.5,.070,
             r"Crossover: $CR_y^{0\to1}(\pi/2)$ then $CR_y^{1\to0}(\pi/2)$  ·  Mutation: $R_x^{2}(\pi/2)$",
             ha="center",fontsize=10)
    fig.text(.5,.039,
             "Example selected qubits; crossover and mutation are each applied with probability 0.5.",
             ha="center",fontsize=9,color="#555555")
    for extension in ["png","svg","pdf"]:
        fig.savefig(output_path(extension, 'aeqga_circuit_diagram.'+extension),dpi=240,facecolor="white")
    plt.close(fig)
    print("Saved aeqga_circuit_diagram.png, .svg, and .pdf")


if __name__ == "__main__":
    main()
