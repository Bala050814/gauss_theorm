import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define symbols
x, y, z = sp.symbols('x y z')

st.set_page_config(page_title="Gauss Divergence Theorem", layout="wide")
st.title("Gauss Divergence Theorem Calculator")

st.markdown("""
This calculator verifies the **Gauss Divergence Theorem**:

### ∯_S **F** · dS = ∭_V div(**F**) dV

You provide a vector field **F(x, y, z)** and a box volume, and this app computes the **divergence** and the **volume integral** side of the theorem. It also shows a 3D **vector field plot**.
""")

# Input vector field
with st.expander("🔢 Input Vector Field Components"):
    Fx_input = st.text_input("Fₓ(x, y, z):", "x*y")
    Fy_input = st.text_input("Fᵧ(x, y, z):", "y*z")
    Fz_input = st.text_input("F𝓏(x, y, z):", "z*x")

# Input bounds
st.subheader("📦 Volume Bounds (Rectangular Box)")

col1, col2, col3 = st.columns(3)
with col1:
    x1 = st.number_input("x lower", value=-1.0)
    x2 = st.number_input("x upper", value=1.0)
with col2:
    y1 = st.number_input("y lower", value=-1.0)
    y2 = st.number_input("y upper", value=1.0)
with col3:
    z1 = st.number_input("z lower", value=-1.0)
    z2 = st.number_input("z upper", value=1.0)

compute_btn = st.button("🧮 Compute")

if compute_btn:
    try:
        # Parse vector field components
        Fx = sp.sympify(Fx_input)
        Fy = sp.sympify(Fy_input)
        Fz = sp.sympify(Fz_input)

        # Compute divergence
        div_F = sp.diff(Fx, x) + sp.diff(Fy, y) + sp.diff(Fz, z)
        st.latex(r"\text{Divergence: } \nabla \cdot \vec{F} = " + sp.latex(div_F))

        # Compute volume integral of divergence
        vol_integral = sp.integrate(div_F, (x, x1, x2), (y, y1, y2), (z, z1, z2))
        st.success(f"✅ Volume Integral (∭ div F dV): {vol_integral.evalf()}")

        # Vector field plot
        st.subheader("📊 Vector Field Visualization (3D Animated Quiver Plot)")

        grid_size = 5
        X_vals = np.linspace(x1, x2, grid_size)
        Y_vals = np.linspace(y1, y2, grid_size)
        Z_vals = np.linspace(z1, z2, grid_size)
        X, Y, Z = np.meshgrid(X_vals, Y_vals, Z_vals)

        # Convert symbolic expressions to numerical functions
        Fx_func = sp.lambdify((x, y, z), Fx, "numpy")
        Fy_func = sp.lambdify((x, y, z), Fy, "numpy")
        Fz_func = sp.lambdify((x, y, z), Fz, "numpy")

        U = Fx_func(X, Y, Z)
        V = Fy_func(X, Y, Z)
        W = Fz_func(X, Y, Z)

        # Plotting
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        quiver = ax.quiver(X, Y, Z, U, V, W, length=0.2, normalize=True, color='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Vector Field: F(x, y, z)', fontsize=14)

        # Animation function
        def update_quiver(num, X, Y, Z, U, V, W, quiver):
            quiver.remove()
            U_new = U * np.cos(num * np.pi / 50)  # Change over time
            V_new = V * np.sin(num * np.pi / 50)  # Change over time
            W_new = W * np.cos(num * np.pi / 50)  # Change over time
            return ax.quiver(X, Y, Z, U_new, V_new, W_new, length=0.2, normalize=True, color='blue')

        ani = FuncAnimation(fig, update_quiver, frames=100, fargs=(X, Y, Z, U, V, W, quiver), interval=100)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"🚫 Error: {e}")
