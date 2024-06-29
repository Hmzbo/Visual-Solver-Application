import os
import time

import google.generativeai as genai
import streamlit as st

from VisuaLLMOpti import LLMSolver, OptProbVisualSolver, TexPixToPython

st.set_page_config(
    layout="wide",
    page_title="VisualLLMOpti",
    page_icon="📋",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

st.title("Visual Optimization Solver")
st.subheader(":red[Solve Problems & Generate Insightful Reports]")


@st.cache_resource
def verify_gemini_key(key) -> tuple[genai.GenerativeModel, genai.GenerativeModel]:
    """
    A function that verifies the Gemini key by configuring the API key,
    creating two GenerativeModel instances, test the key, setting the API key
    in the environment, and returning the two GenerativeModel instances.

    Args:
        key (str): The Gemini API key to be verified.

    Returns:
        Tuple[genai.GenerativeModel, genai.GenerativeModel]: A tuple containing
        two Gemini instances (v1 and v1.5), if the verification is successful.

    Raises:
        Exception: If an error occurs during the verification process, an exception is raised.
    """
    try:
        genai.configure(api_key=gemini_key)
        gemini_15_model = genai.GenerativeModel("gemini-1.5-pro")
        gemini_1_model = genai.GenerativeModel("gemini-pro")
        response = gemini_1_model.generate_content("hello")
        os.environ["API_KEY"] = gemini_key
        return gemini_1_model, gemini_15_model
    except Exception as e:
        st.exception(e)


with st.sidebar:
    # Initialize the session state
    if "img_obj_func_string" not in st.session_state:
        st.session_state.img_obj_func_string = ""
    if "img_const_func_string" not in st.session_state:
        st.session_state.img_const_func_string = ""
    if "minx" not in st.session_state:
        st.session_state.minx = None
    if "miny" not in st.session_state:
        st.session_state.miny = None
    if "maxx" not in st.session_state:
        st.session_state.maxx = None
    if "maxy" not in st.session_state:
        st.session_state.maxy = None
    if "x_precision" not in st.session_state:
        st.session_state.x_precision = 100
    if "y_precision" not in st.session_state:
        st.session_state.y_precision = 100
    if "solv_precision_factor" not in st.session_state:
        st.session_state.solv_precision_factor = 10
    if "xmin_buff" not in st.session_state:
        st.session_state.xmin_buff = None
    if "xmax_buff" not in st.session_state:
        st.session_state.xmax_buff = None
    if "ymin_buff" not in st.session_state:
        st.session_state.ymin_buff = None
    if "ymax_buff" not in st.session_state:
        st.session_state.ymax_buff = None
    if "zmin_buff" not in st.session_state:
        st.session_state.zmin_buff = None
    if "zmax_buff" not in st.session_state:
        st.session_state.zmax_buff = None
    if "extrusion_buff" not in st.session_state:
        st.session_state.extrusion_buff = 10.0
    if "range_x" not in st.session_state:
        st.session_state.range_x = 1.0
    if "range_y" not in st.session_state:
        st.session_state.range_y = 1.0
    if "val_ini_x" not in st.session_state:
        st.session_state.val_ini_x = None
    if "val_ini_y" not in st.session_state:
        st.session_state.val_ini_y = None

    # Buttons to load examples
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Exp 1", use_container_width=True):
            st.session_state.img_obj_func_string = "x**2+y**2"
            st.session_state.img_const_func_string = "x*y -1"
            st.session_state.minx = st.session_state.miny = -10.0
            st.session_state.maxx = st.session_state.maxy = 10.0
            st.session_state.x_precision = st.session_state.y_precision = 100
            st.session_state.solv_precision_factor = 15
            st.session_state.xmin_buff = st.session_state.ymin_buff = (
                st.session_state.zmin_buff
            ) = 3.0
            st.session_state.xmax_buff = st.session_state.ymax_buff = (
                st.session_state.zmax_buff
            ) = 3.0
            st.session_state.range_x = st.session_state.range_y = 20
            st.session_state.val_ini_x = 1.0
            st.session_state.val_ini_y = 4.0
    with col2:
        if st.button("Load Exp 2", use_container_width=True):
            st.session_state.img_obj_func_string = "-3*np.exp(x)+y*x**2"
            st.session_state.img_const_func_string = "y - np.exp(x)"
            st.session_state.minx = st.session_state.miny = -10.0
            st.session_state.maxx = st.session_state.maxy = 10.0
            st.session_state.x_precision = st.session_state.y_precision = 100
            st.session_state.solv_precision_factor = 30
            st.session_state.xmin_buff = 10.0
            st.session_state.ymin_buff = 5.0
            st.session_state.zmin_buff = 10.0
            st.session_state.xmax_buff = 10.0
            st.session_state.ymax_buff = 5.0
            st.session_state.zmax_buff = 20.0
            st.session_state.val_ini_x = -2.0
            st.session_state.val_ini_y = 8.0
            st.session_state.extrusion_buff = 20.0
    if st.button("Load Exp 3", use_container_width=True):
        st.session_state.img_obj_func_string = "8*np.exp(-x**2-y**2)*\
            (0.1+x*(y-0.5))"
        st.session_state.img_const_func_string = "np.sin((x-3)*(y-3))"
        st.session_state.minx = st.session_state.miny = -10.0
        st.session_state.maxx = st.session_state.maxy = 10.0
        st.session_state.x_precision = st.session_state.y_precision = 100
        st.session_state.solv_precision_factor = 20
        st.session_state.xmin_buff = st.session_state.ymin_buff = (
            st.session_state.zmin_buff
        ) = 3.0
        st.session_state.xmax_buff = st.session_state.ymax_buff = 3.0
        st.session_state.zmax_buff = 6.0
        st.session_state.val_ini_x = -0.1
        st.session_state.val_ini_y = -0.5
    st.divider()

    st.write("👀 :orange[Check 'How to use?' page]")
    st.header("Settings")

    gemini_key = st.text_input(
        "gemini-pro key",
        type="password",
        help="You can get your API for free! (See How does it work page)",
    )
    if gemini_key:
        result = verify_gemini_key(gemini_key)
        if result is not None:
            gemini_1_model, gemini_15_model = result
        else:
            st.error("Failed to verify the gemini key")

    st.subheader("Upload image or input manually:", divider=True)
    uploaded_image = st.file_uploader(
        "Upload problem statement image",
        type=["png", "jpg"],
        accept_multiple_files=False,
        help="Problem statement must be in LaTeX format to work properly.",
    )

    if st.button("Extract"):
        if uploaded_image is not None:
            try:
                converter = TexPixToPython(gemini_15_model)
            except:
                st.exception(Exception("Make sure you input your API KEY"))
            extracted_functions_dic = converter.convert_tex_to_python(uploaded_image)
            st.session_state.img_obj_func_string = extracted_functions_dic["obj_func"]
            st.session_state.img_const_func_string = extracted_functions_dic[
                "constraint_func"
            ]

    obj_func_string = st.text_input(
        "Objective function",
        value=st.session_state.img_obj_func_string,
        help="Must be respecting Python with NumPy syntax.",
        placeholder="Example: x**2+np.exp(y)",
    )
    const_func_string = st.text_input(
        "Equality Constraint function",
        value=st.session_state.img_const_func_string,
        help="Must be respecting Python with NumPy syntax.",
        placeholder="Example: x-y+np.pi",
    )

    st.subheader("Enter search domain", divider=True)

    col1, col2 = st.columns(2)
    with col1:
        minx = st.number_input(
            "Min x value", value=st.session_state.minx, placeholder="-10"
        )
    with col2:
        maxx = st.number_input(
            "Max x value", value=st.session_state.maxx, placeholder="10"
        )
    if maxx and minx:
        st.session_state.range_x = maxx - minx
        x_range_bool = True

    col1, col2 = st.columns(2)
    with col1:
        miny = st.number_input(
            "Min y value", value=st.session_state.miny, placeholder="-10"
        )
    with col2:
        maxy = st.number_input(
            "Max y value", value=st.session_state.maxy, placeholder="10"
        )
    if maxy and miny:
        st.session_state.range_y = maxy - miny
    st.subheader("Visual solver parameters", divider=True)

    col1, col2 = st.columns(2)
    with col1:
        x_precision = st.slider(
            "X precision",
            10,
            300,
            value=st.session_state.x_precision,
            step=10,
            help="Numercial precision for mesh construction",
        )
    with col2:
        y_precision = st.slider(
            "Y precision",
            10,
            300,
            value=st.session_state.y_precision,
            step=10,
            help="Numercial precision for mesh construction",
        )
    solv_precision_factor = st.slider(
        "Precision multiplier",
        1,
        100,
        value=st.session_state.solv_precision_factor,
        step=1,
        help="Numercial precision multiplier for visual solver",
    )

    st.subheader("Visualizations parameters", divider=True)
    col1, col2 = st.columns(2)

    with col1:
        xmin_buff = st.number_input(
            "X min buff",
            min_value=0.0,
            value=st.session_state.xmin_buff,
            placeholder="10",
        )
    with col2:
        xmax_buff = st.number_input(
            "X max buff",
            min_value=0.0,
            value=st.session_state.xmax_buff,
            placeholder="10",
        )

    col1, col2 = st.columns(2)
    with col1:
        ymin_buff = st.number_input(
            "Y min buff",
            min_value=0.0,
            value=st.session_state.ymin_buff,
            placeholder="10",
        )
    with col2:
        ymax_buff = st.number_input(
            "Y max buff",
            min_value=0.0,
            value=st.session_state.ymax_buff,
            placeholder="10",
        )

    col1, col2 = st.columns(2)
    with col1:
        zmin_buff = st.number_input(
            "Z min buff",
            min_value=0.0,
            value=st.session_state.zmin_buff,
            placeholder="10",
        )
    with col2:
        zmax_buff = st.number_input(
            "Z max buff",
            min_value=0.0,
            value=st.session_state.zmax_buff,
            placeholder="10",
        )

    extrusion_buff = st.number_input(
        "Z extrusion buff",
        min_value=0.0,
        value=st.session_state.extrusion_buff,
        placeholder="10",
    )


settings = {
    "obj_func": obj_func_string,
    "constraint_func": const_func_string,
    "x_domain": (minx, maxx),
    "y_domain": (miny, maxy),
    "x_precision": x_precision,
    "y_precision": y_precision,
    "precision_multip": solv_precision_factor,
    "x_buff": (xmin_buff, xmax_buff),
    "y_buff": (ymin_buff, ymax_buff),
    "z_buff": (zmin_buff, zmax_buff),
    "extrusion_buff": extrusion_buff,
}


tab1, tab2, tab3 = st.tabs(
    ["How to use?", "Solve Optimization Problems", "Perform Projected Gradient Descent"]
)

with tab1:
    st.subheader("How to use?", divider=True)
    with st.expander("You need a Gemini API key to generate a report"):
        st.markdown(
            """You can get a Gemini API key for free. Here are the steps to \
                obtain it:
1. **Visit Google AI Studio**: Go to the [Google AI Studio](https://ai.google.\
    dev/gemini-api/docs/api-key) website.
2. **Sign In**: Sign in with your Google account.
3. **Create API Key**: Under the API keys section, click on the “Create API \
    key in new project” button.
4. **Copy the API Key**: Once generated, copy the API key.
"""
        )
    st.markdown(
        r"""
1. Enter your Gemini API key on the designated field in the sidebar.
2. You have two options:
    - Use the image uploader to upload the problem statement and then use the
    Extract button to extract it.
    - Enter the problem statement manually in the corresponding fields.
3. Enter the search domain limits $[x_{\text{min}}, x_{\text{max}}]$, and
$[y_{\text{min}}, y_{\text{max}}]$.
4. Precision:
    - Select the precision to use along the X-axis and Y-axis to create the
    mesh. Higher values lead to smoother plots but higher memory usage and
    longer computation time.
    - Select the precision multiplier that will be used in computing the
    numerical solution (Solver_recision = Precision x Precision Multiplier),
    higher values don't affect computation time as much as Precision values.
5. Enter the buff values to use along each axis when plotting results.
6. Head over to the "Solve optimization problem" page to solve the problem
graphically and generate a report.
7. You can also solve the problem using the Projected gradient descent method.
However, it's convergence to a global optimum isn't guarenteed.
"""
    )
    st.info(
        "If you don't want to use Gemini, you can still use the visual solver \
        and the PGD tool."
    )

with tab2:
    with st.expander("Steps to solve this type of problems."):
        st.subheader(
            "Solving Optimization Problems with Equality Constraint Using the \
                Lagrangian Method",
            divider=True,
        )
        st.write(
            """Optimization problems with one equality constraint can be
            effectively solved using the Lagrangian method.
                This approach introduces a Lagrange multiplier to incorporate
                the constraint into the objective function,
                transforming the problem into an unconstrained optimization
                scenario."""
        )
        st.markdown(
            """**Problem formulation**:\\
    Consider the optimization problem:"""
        )
        st.latex(
            r"""\min\limits_{(x,y)\in\mathbb{R}^2}f(x,y)\quad \text{subject
            to}\quad g(x,y)=0"""
        )
        st.markdown(
            "Here, $f(x,y)$ is called the **objective function**, \
                and $g(x,y)=0$ represents the **equality constraint**."
        )
        st.markdown(
            """**Lagrangian function**:\\
    The Lagrangian function $\mathcal{L}(x,y,\lambda)$ is defined as: """
        )
        st.latex(r"""\mathcal{L}(x,y,\lambda)=f(x,y) + \lambda g(x,y)""")
        st.markdown("where $\lambda$ is the Lagrange multiplier.")
        st.markdown(
            r"""**Regularity Condition (Constraint Qualification)**:\
    A point $(x^*,y^*)$ satisfies the regularity condition (or constraint
    qualification) if $\nabla g(x^*,y^*)\neq 0$ . This ensures the constraint
    is active and properly influences the solution.\
    **First Order Necessary Condition of Optimality**:\
    For $(x^*,y^*)$ to be an optimal solution, it must satisfy the following
    system of equations:"""
        )
        st.latex(
            r"""\begin{cases}\nabla_{xy}\mathcal{L}(x^*,y^*,\lambda^*) =
            \nabla_{xy} f(x^*,y^*) + \lambda \nabla_{xy} g(x^*,y^*) = 0\\
                \nabla_\lambda\mathcal{L}(x^*,y^*,\lambda^*) = g(x^*,y^*) =
                0 \end{cases}"""
        )
        st.write(
            "These conditions ensure that the gradients of the objective \
                function and the constraint are aligned at the optimal point, \
                scaled by the Lagrange multiplier."
        )
        st.markdown(
            r""" **Second-Order Necessary Condition of Optimality**:\
    The Hessian matrix $\text{Hess}_{xy}(\mathcal{L})$ (the matrix of
    second-order partial derivatives) of the Lagrangian at the point $(x^*,y^*,
    \lambda^*)$ must be positive semi-definite."""
        )
        st.markdown(
            r""" **Second-Order Sufficient Condition of Optimality**:\
    The Hessian of the Lagrangian, $\text{Hess}_{xy}(\mathcal{L})$, evaluated
    at $(x^*,y^*,\lambda^*)$, should be positive definite
    when projected onto the feasible directions:"""
        )
        st.latex(
            r""" d^T\text{Hess}_{xy}(\mathcal{L})(x^*,y^*,\lambda^*)d>0,\quad
            \forall d\neq 0, \nabla g(x^*,y^*)^Td=0 """
        )
    col1, col2, col3, col4 = st.columns(4)
    with col2:
        solve_report_btn = st.button(
            "Solve visually & Generate report", use_container_width=True
        )
    with col3:
        solve_btn = st.button("Solve visually", use_container_width=True)

    if solve_report_btn or solve_btn:
        prog_bar = st.progress(0, text="Initializing..")
        opt = OptProbVisualSolver(
            obj_func_string=settings["obj_func"],
            constraint_func_string=settings["constraint_func"],
            x_domain=settings["x_domain"],
            y_domain=settings["y_domain"],
            x_precision=settings["x_precision"],
            y_precision=settings["y_precision"],
            solv_precision_factor=settings["precision_multip"],
        )
        prog_bar.progress(10, text="Running feasability tests..")
        opt._test_func_domains()
        fg0_intersection_pts, g_contour_zero_values = opt._check_feasablity()
        num_opt_sol = opt._numerical_solver(initial_guess=(minx, miny))

        prog_bar.progress(20, text="Solving graphically..")
        results = opt.solve_graphically(
            omit_num_res=False,
            z_contour_tile_buff=settings["extrusion_buff"],
            round_deci=5,
        )
        if isinstance(results, dict):
            num_sol_res_dic = results
            st.write(num_sol_res_dic)
            prog_bar.empty()
        elif results is None:
            st.write(
                "Couldn't find minima! Solution doesn't exist. Or 'Precision \
                    multiplier' is too low."
            )
            prog_bar.empty()
        else:
            fig_sol, figs = opt.visualize(
                results,
                "sol",
                z_buff=settings["z_buff"],
                x_buff=settings["x_buff"],
                y_buff=settings["y_buff"],
            )
            fig_g, figs = opt.visualize(
                results,
                "g",
                z_buff=settings["z_buff"],
                x_buff=settings["x_buff"],
                y_buff=settings["y_buff"],
            )
            fig_g0, figs = opt.visualize(
                results,
                "g0",
                z_buff=settings["z_buff"],
                x_buff=settings["x_buff"],
                y_buff=settings["y_buff"],
            )

        if solve_report_btn:
            prog_bar.progress(50, text="Generating report..")
            LLM_Solver = LLMSolver(gemini_15_model)
            LLM_RESULTS = LLM_Solver.generate_report(
                settings["obj_func"], settings["constraint_func"], prog_bar
            )

            def stream_data_step_1():
                for word in LLM_RESULTS[0].split(" "):
                    yield word + " "
                    time.sleep(0.02)

            def stream_data_step_2():
                for word in LLM_RESULTS[1].split(" "):
                    yield word + " "
                    time.sleep(0.02)

            def stream_data_step_3():
                for word in LLM_RESULTS[2].split(" "):
                    yield word + " "
                    time.sleep(0.02)

            def stream_data_step_4():
                for word in LLM_RESULTS[3].split(" "):
                    yield word + " "
                    time.sleep(0.02)

            def stream_data_step_5():
                for word in LLM_RESULTS[4].split(" "):
                    yield word + " "
                    time.sleep(0.02)

            def stream_data_step_6():
                for word in LLM_RESULTS[5].split(" "):
                    yield word + " "
                    time.sleep(0.02)

            def stream_data_step_7():
                for word in LLM_RESULTS[6].split(" "):
                    yield word + " "
                    time.sleep(0.02)

        prog_bar.progress(100, text="Done!")
        time.sleep(0.5)
        prog_bar.empty()

        if solve_report_btn:
            st.write_stream(stream_data_step_1)

        col1, col2, col3, col4 = st.columns([0.1, 0.4, 0.4, 0.1])
        with col2:
            st.plotly_chart(fig_g, theme="streamlit", use_container_width=False)
        with col3:
            st.plotly_chart(fig_g0, theme="streamlit", use_container_width=False)

        if solve_report_btn:
            st.write_stream(stream_data_step_2)
            st.write_stream(stream_data_step_3)
            st.write_stream(stream_data_step_4)
            st.write_stream(stream_data_step_5)
            st.write_stream(stream_data_step_6)

        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
        with col2:
            st.plotly_chart(fig_sol, theme="streamlit", use_container_width=False)

        if solve_report_btn:
            st.write_stream(stream_data_step_7)

with tab3:
    with st.expander("Projected Gradient Descent"):
        st.markdown(
            """**Projected Gradient Descent (PGD)** is an optimization algorithm
              used to solve constrained optimization problems. It extends the
              standard gradient descent method by incorporating a projection
              step to ensure that the iterates remain within a feasible set
              defined by the constraints.

Here’s a brief overview:
- **Gradient Descent Step:** Compute the gradient of the objective function and
take a step in the direction of the negative gradient to minimize the function.
- **Projection Step:** Project the resulting point back onto the feasible set if
 it falls outside the constraints.
This method is particularly useful in scenarios where the solution must satisfy
certain constraints, such as non-negativity or bounded regions. PGD is widely
used in machine learning and signal processing applications."""
        )

    st.warning(
        ":blue[This method is highly sensitive to the starting point, and \
        doesn't guarentee convergance to the optimal solution.]"
    )

    st.subheader("PGD parameters", divider=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if minx is not None and st.session_state.val_ini_x is None:
            st.session_state.val_ini_x = minx + st.session_state.range_x / 2
        if miny is not None and st.session_state.val_ini_y is None:
            st.session_state.val_ini_y = miny + st.session_state.range_y / 2

        x_ini = st.number_input(
            "Starting point x",
            min_value=minx,
            max_value=maxx,
            value=st.session_state.val_ini_x,
        )
        y_ini = st.number_input(
            "Starting point y",
            min_value=miny,
            max_value=maxy,
            value=st.session_state.val_ini_y,
        )

    with col2:
        max_iter = st.slider(
            "Max iter",
            100,
            2000,
            value=1000,
            step=10,
            help="Maximum number of iteration",
        )
        alpha = st.number_input(
            "Alpha", 0.0001, value=0.02, help="Gradient descent step size", format="%e"
        )
    with col3:
        tol = st.number_input(
            "Convergence tolerance",
            min_value=1e-10,
            max_value=1e-5,
            value=1e-8,
            format="%e",
        )
    with col4:
        grad_field_scale = st.number_input(
            "Grad field scale",
            min_value=0.001,
            max_value=1.0,
            value=0.02,
            help="Gradient field plotting scale",
            format="%f",
        )

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col2:
        run_pgd = st.button("Run & Visualize", use_container_width=True)

    if run_pgd:
        prog_bar = st.progress(0, text="Initializing..")
        opt = OptProbVisualSolver(
            obj_func_string=settings["obj_func"],
            constraint_func_string=settings["constraint_func"],
            x_domain=settings["x_domain"],
            y_domain=settings["y_domain"],
            x_precision=settings["x_precision"],
            y_precision=settings["y_precision"],
            solv_precision_factor=settings["precision_multip"],
        )
        steps = opt.solve_PGD(
            prog_bar, init_pt=(x_ini, y_ini), max_iter=max_iter, alpha=alpha, tol=tol
        )
        fig_pdg = opt.visualize_PGD(
            steps,
            x_buff=settings["x_buff"],
            y_buff=settings["y_buff"],
            z_buff=settings["z_buff"],
            grad_field_scale=grad_field_scale,
        )
        prog_bar.progress(100, text="Done")
        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
        with col2:
            st.plotly_chart(fig_pdg, theme="streamlit", use_container_width=True)
