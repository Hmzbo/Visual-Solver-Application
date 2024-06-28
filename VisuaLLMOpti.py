import time
from typing import Union

import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objs as go
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from PIL import Image
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from tqdm import tqdm


class OptProbVisualSolver:
    def __init__(
        self,
        obj_func_string: str,
        constraint_func_string: str,
        x_domain: Union[list, tuple],
        y_domain: Union[list, tuple],
        x_precision: Union[None, int] = None,
        y_precision: Union[None, int] = None,
        solv_precision_factor: int = 7,
    ) -> None:
        """Initializes the OptProbVisualSolver with the specified objective and
        constraint function strings, x and y domains, optional x and y
        precisions, and the solver precision factor.

        Parameters:
            - obj_func_string (str): The objective function string.
            - constraint_func_string (str): The constraint function string.
            - x_domain (Union[list, tuple]): The domain for the x variable.
            - y_domain (Union[list, tuple]): The domain for the y variable.
            - x_precision (Union[None, int], optional): The numercial precision
             for the x variable. Defaults to None.
            - y_precision (Union[None, int], optional): The numerical precision
             for the y variable. Defaults to None.
            - solv_precision_factor (int): The solver numercial precision
            factor.

        Returns:
            None
        """

        # Check input variables type
        if not isinstance(x_domain, (list, tuple)):
            raise ValueError("'x_domain' must be a List or a Tuple.")
        if not isinstance(y_domain, (list, tuple)):
            raise ValueError("'y_domain' must be a List or a Tuple.")
        if not isinstance(obj_func_string, str):
            raise ValueError("'obj_func_string' must be a String.")
        if not isinstance(constraint_func_string, str):
            raise ValueError("'constraint_func_string' must be a String.")
        if x_precision:
            if not isinstance(x_precision, int):
                raise ValueError("'x_precision' must be an integer.")
            if x_precision not in range(50, 10001):
                raise ValueError("'x_precision' must be in [50,10000].")
        if y_precision:
            if not isinstance(y_precision, int):
                raise ValueError("'y_precision' must be an integer.")
            if y_precision not in range(50, 10001):
                raise ValueError("'y_precision' must be in [50,10000].")

        # Check inut variables values
        try:
            compile(obj_func_string, "<string>", "exec")
        except SyntaxError as e:
            raise ValueError(
                f"Invalid Python syntax provided for 'obj_func_string'. Error \
                    details: {e.msg}."
            ) from e
        try:
            compile(constraint_func_string, "<string>", "exec")
        except SyntaxError as e:
            raise ValueError(
                f"Invalid string Python syntax provided for 'obj_func_string'. \
                    Error details: {e.msg}."
            ) from e
        if len(x_domain) != 2 or len(y_domain) != 2:
            raise ValueError("Input variables 'x' and 'y' must be of length 2.")

        x_range = x_domain[1] - x_domain[0]
        y_range = y_domain[1] - y_domain[0]
        if x_precision is None:
            x_precision = self.get_precision_from_range(x_range)
        if y_precision is None:
            y_precision = self.get_precision_from_range(y_range)

        # Set OptProbVisualSolver attributes
        self.obj_func_string = obj_func_string
        self.constraint_func_string = constraint_func_string
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.x_range = x_range
        self.y_range = y_range
        self.x_precision = x_precision
        self.y_precision = y_precision
        self.x_solv_precision = min(x_precision * solv_precision_factor, 5000)
        self.y_solv_precision = min(y_precision * solv_precision_factor, 5000)

    # Define objective function
    def objective_function(self, variables: Union[list, tuple, np.ndarray]):
        """Evaluates the objective function using the given variables.

        Parameters:
            variables (Union[list, tuple, np.ndarray]): The variables to be \
                used in the evaluation. It can be a list, tuple, or numpy \
                    array containing the values of 'x' and 'y'.

        Returns:
            float: The value of the objective function evaluated at the given \
                variables.
        """
        x, y = variables
        return eval(self.obj_func_string)

    # Define constraint function
    def constraint_function(self, variables: Union[list, tuple, np.ndarray]):
        """Evaluates the constraint function using the given variables.

        Parameters:
            variables (Union[list, tuple, np.ndarray]): The variables to be \
                used in the evaluation.
                It can be a list, tuple, or numpy array containing the values \
                    of 'x' and 'y'.

        Returns:
            The result of evaluating the constraint function with the given \
                variables.
        """
        x, y = variables
        return eval(self.constraint_func_string)

    def _test_functions(self):
        """Test the objective and constraint functions by evaluating them at
        the first point in the x and y domains.

        Raises:
            Exception: If the objective or constraint function uses variables
                'x' and 'y' with incorrect domains.
            Exception: If the objective or constraint function returns NaN for
                the first point in the x and y domains.
        """
        try:
            zf0 = self.objective_function((self.x_domain[0], self.y_domain[0]))
        except Exception as e:
            raise Exception(
                "Ensure that the objective function uses variables 'x' and 'y' \
                    with correct domains ('x_domain[0]' & 'y_domain[0]' must \
                        be in objective function domain)."
            ) from e
        try:
            zg0 = self.constraint_function((self.x_domain[0], self.y_domain[0]))
        except Exception as e:
            raise Exception(
                "Ensure that the constraint function uses variables 'x' and 'y'\
                      with correct domains ('x_domain[0]' & 'y_domain[0]' must\
                          be in constraint function domain)."
            ) from e
        if np.isnan(zf0):
            raise Exception(
                "Ensure that the objective function uses variables 'x' and 'y'\
                      with correct domains ('x_domain[0]' & 'y_domain[0]' must\
                          be in objective function domain)."
            )
        if np.isnan(zg0):
            raise Exception(
                "Ensure that the constraint function uses variables 'x' and 'y'\
                      with correct domains ('x_domain[0]' & 'y_domain[0]' must\
                          be in constraint function domain)."
            )

    @staticmethod
    def get_precision_from_range(range):
        """Calculate the precision based on the given range.

        Parameters:
            range (int): The range value.

        Returns:
            int: The calculated precision value.

        This static method calculates the precision based on the given range.
        It checks the range value and assigns a corresponding precision
        value. If the range is less than 10, the precision is set to
        100. If the range is between 10 and 100, the precision is
        set to 300. If the range is between 100 and 1000, the
        precision is set to 500. Otherwise, the precision
        is set to 1000.

        Example:
            >>> get_precision_from_range(5)
            100
            >>> get_precision_from_range(20)
            300
            >>> get_precision_from_range(500)
            500
            >>> get_precision_from_range(1001)
            1000
        """
        if range < 10:
            precision = 100
        elif range < 100:
            precision = 300
        elif range < 1000:
            precision = 500
        else:
            precision = 1000
        return precision

    def _check_feasablity(self):
        """
        Checks the feasibility of the model by performing the following steps:
        1. Generate values for x and y within the specified domains.
        2. Calculate the values of the constraint function g(x, y) over the
            grid created by the generated values.
        3. Find the contour level corresponding to 0 and extract the vertices
            of the contour path.
        4. If the contour path has no vertices, raise an exception.
        5. Iterate over the vertices and evaluate the objective function for
            each vertex.
        6. If the evaluation is not NaN, add the vertex to the intersection
            points list.
        7. If the intersection points list is not empty, return the
            intersection points and the contour vertices.
        8. Otherwise, raise an exception indicating that the model has no
            solution.

        Returns:
            intersection_pts (list): A list of intersection points between the
                feasible set and the objective function domain.
            g_contour_zero_values (ndarray): The vertices of the contour path
                corresponding to the contour level 0.

        Raises:
            Exception: If the feasible set is empty or discrete.
            Exception: If the model has no solution.
        """
        # Generate values for x and y
        x = np.linspace(self.x_domain[0], self.x_domain[1], self.x_solv_precision)
        y = np.linspace(self.y_domain[0], self.y_domain[1], self.y_solv_precision)
        # Create a grid of x and y values
        x_grid, y_grid = np.meshgrid(x, y)
        # Calculate the values of the function g(x, y) over the grid
        z_g = self.constraint_function([x_grid, y_grid])
        g_contour = plt.contour(
            x_grid, y_grid, z_g, levels=[0]
        )  # Contour level corresponding to 0
        g_contour_zero_values = g_contour.get_paths()[0].vertices
        plt.close()
        if g_contour.get_paths()[0].vertices.shape[0] == 0:
            raise Exception(
                "The feasable set is either empty or descrete[no supported yet]\
                    ! Check model statement."
            )

        intersection_pts = []

        for i in range(g_contour_zero_values.shape[0]):
            try:
                f_guess = self.objective_function(g_contour_zero_values[i, :])
                if ~np.isnan(f_guess):
                    intersection_pts.append(g_contour_zero_values[i, :])
            except:
                pass
        if len(intersection_pts) != 0:
            return intersection_pts, g_contour_zero_values
        else:
            raise Exception(
                "Model has no solution! The feasable set has no intersection \
                    with the objective function domain."
            )

    def _numerical_solver(self, initial_guess):
        """This function solves an optimization problem using numerical
        methods. It takes an initial guess for the solution and defines the
        equality constraint for the problem. The optimization problem is then
        solved using the `minimize` function from the `scipy.optimize` module.
        The result of the optimization problem is returned as an
        `OptimizeResult` object, which includes the optimal solution and other
        information.

        Args:
            initial_guess (ndarray): The initial guess for the solution.

        Returns:
            OptimizeResult: The result of the optimization problem, including
                the optimal solution and other information.

        Note:
            The maximum number of iterations used in `scipy.optimize` is set
                to 1000.

        Example:
            >>> initial_guess = np.array([1.0, 2.0])
            >>> result = _numerical_solver(initial_guess)
            >>> print(result.x)
            [0.5 1.5]
        """
        # Define the equality constraint
        eq_cons = {"type": "eq", "fun": self.constraint_function}
        # Solve the optimization problem
        result = minimize(
            self.objective_function,
            initial_guess,
            constraints=eq_cons,
            options={"maxiter": 1000},
        )
        # Optimal solution
        return result

    def _get_plotly_fig(self, plot_type: str, args: list, **kwargs):
        """This function generates a plotly figure object of the specified
        type. The type of plot can be either "surface" or any other value. The
        function takes a list of arguments that depend on the type of plot. If
        the plot type is "surface", the function generates a surface plot using
        the provided function and the specified x and y ranges. If the plot
        type is not "surface", the function generates a 3D scatter plot using
        the provided x, y, and z coordinates. The function returns the
        generated plotly figure object or a tuple containing the figure object
        and the z-coordinates of the data points.

        Args:
            - plot_type (str): The type of plot to generate. Can be either
                "surface" or any other value.
            - args (list): A list of arguments required to generate the plot.
                The length and contents of the list depend on the value of
                    `plot_type`.
                - If `plot_type` is "surface" and `len(args)` is 5:
                    - `func` (callable): A function that takes a 2D array of
                        x and y values and returns a 2D array of z values.
                    - `xmin` (float): The minimum value of the x-axis.
                    - `xmax` (float): The maximum value of the x-axis.
                    - `ymin` (float): The minimum value of the y-axis.
                    - `ymax` (float): The maximum value of the y-axis.
                - If `plot_type` is "surface" and `len(args)` is not 5:
                    - `x` (array-like): The x-coordinates of the data points.
                    - `y` (array-like): The y-coordinates of the data points.
                    - `z` (array-like): The z-coordinates of the data points.
                - If `plot_type` is not "surface":
                    - `x` (array-like): The x-coordinates of the data points.
                    - `y` (array-like): The y-coordinates of the data points.
                    - `z` (array-like): The z-coordinates of the data points.
            **kwargs: Additional keyword arguments to pass to the plotly figure
                  constructor.

        Returns:
            - tuple or plotly.graph_objs._figure.Figure:
                - If `plot_type` is "surface" and `len(args)` is 5:
                    - `surface` (plotly.graph_objs.Surface): The surface plot
                            object.
                    - `z` (array-like): The z-coordinates of the data points.
                - If `plot_type` is "surface" and `len(args)` is not 5:
                    - `surface` (plotly.graph_objs.Surface): The surface plot
                            object.
                - If `plot_type` is not "surface":
                    - `scatter` (plotly.graph_objs.Scatter3d): The 3D scatter
                            plot object.

        Note:
            The function uses the `np.linspace`, `np.meshgrid`, and
            `go.Surface` functions from the `numpy` and `plotly.graph_objs`
            modules, respectively, to generate the surface plot. The
            `go.Scatter3d` function is used to generate the 3D scatter plot.

        Example:
            >>> fig, z = _get_plotly_fig("surface", [lambda x, y: x**2 + y**2, \
-10, 10, -10, 10])
            >>> fig.show()
            >>> fig, z = _get_plotly_fig("surface", [[-10, 10], [-10, 10], \
np.array([[1, 2], [3, 4]])])
            >>> fig.show()
            >>> fig = _get_plotly_fig("scatter3d", [np.array([1, 2, 3]), \
np.array([4, 5, 6]), np.array([7, 8, 9])])
            >>> fig.show()
        """

        if plot_type == "surface":
            if len(args) == 5:
                func, xmin, xmax, ymin, ymax = args
                # Create x, y grid
                x = np.linspace(xmin, xmax, self.x_precision)
                y = np.linspace(ymin, ymax, self.y_precision)
                x_grid, y_grid = np.meshgrid(x, y)
                # Calculate z values using the function
                z = func([x_grid, y_grid])
                # Create the surface plot
                surface = go.Surface(x=x, y=y, z=z, **kwargs)
                return surface, z
            else:
                x, y, z = args
                surface = go.Surface(x=x, y=y, z=z, **kwargs)
                return surface
        else:
            x, y, z = args
            scatter = go.Scatter3d(x=x, y=y, z=z, **kwargs)
            return scatter

    def _get_contour_paths_verticies_and_tiles(
        self, func, xmin, xmax, ymin, ymax, precision_type: str, level=0
    ):
        """Calculates contour paths, vertices, and tiles based on the provided
        function and input parameters.

        Parameters:
            func: The function to calculate z values.
            xmin: The minimum value for the x-axis.
            xmax: The maximum value for the x-axis.
            ymin: The minimum value for the y-axis.
            ymax: The maximum value for the y-axis.
            precision_type: String indicating the precision type ('solv' or
            other).
            level: The contour level (default is 0).

        Returns:
            x_contour: X values of the contour.
            y_contour: Y values of the contour.
            x_contour_list: List of X values for each contour segment.
            y_contour_list: List of Y values for each contour segment.
            x_contour_tile_list: List of tiled X values for each contour
            segment.
            y_contour_tile_list: List of tiled Y values for each contour
            segment.
        """

        # set precision for x and y
        if precision_type == "solv":
            x_precision = self.x_solv_precision
            y_precision = self.y_solv_precision
        else:
            x_precision = self.x_precision
            y_precision = self.y_precision

        # Calculate z values using the function
        x = np.linspace(xmin, xmax, x_precision)
        y = np.linspace(ymin, ymax, y_precision)
        x_grid, y_grid = np.meshgrid(x, y)
        z = func([x_grid, y_grid])

        contour = plt.contour(
            x_grid, y_grid, z, levels=[level]
        )  # Contour level corresponding to level 0
        contour_zero_values = contour.get_paths()[0].vertices
        x_contour = contour_zero_values[:, 0]
        y_contour = contour_zero_values[:, 1]
        plt.close()

        tol = 0.5  # won't work for high range domains for now
        x_contour_tile_list = []
        y_contour_tile_list = []
        x_contour_list = []
        y_contour_list = []
        j = 0
        for i in range(x_contour.shape[0] - 1):
            if (
                np.abs(x_contour[i] - x_contour[i + 1]) > tol
                or np.abs(y_contour[i] - y_contour[i + 1]) > tol
            ):
                x_contour_temp = x_contour[j : i + 1]
                y_contour_temp = y_contour[j : i + 1]
                x_contour_list.append(x_contour_temp)
                y_contour_list.append(y_contour_temp)
                x_contour_temp_tile = np.tile(
                    x_contour_temp, (2 * len(x_contour_temp), 1)
                )
                y_contour_temp_tile = np.tile(
                    y_contour_temp, (2 * len(y_contour_temp), 1)
                )
                x_contour_tile_list.append(x_contour_temp_tile)
                y_contour_tile_list.append(y_contour_temp_tile)
                j = i + 1

        x_contour_tile_list.append(np.tile(x_contour[j:], (2 * len(x_contour[j:]), 1)))
        y_contour_tile_list.append(np.tile(y_contour[j:], (2 * len(y_contour[j:]), 1)))
        x_contour_list.append(x_contour[j:])
        y_contour_list.append(y_contour[j:])

        return (
            x_contour,
            y_contour,
            x_contour_list,
            y_contour_list,
            x_contour_tile_list,
            y_contour_tile_list,
        )

    def _calculate_zintersec_ztiles(
        self, x_contour, y_contour, x_tiles, buff, rounding_deci
    ):
        """Calculates the intersection points of the contour with the objective
        function and generates new z values for each tile.

        Args:
            - x_contour (numpy.ndarray): The x-coordinates of the contour.
            - y_contour (numpy.ndarray): The y-coordinates of the contour.
            - x_tiles (list): A list of x-coordinates for each tile.
            - buff (float): The buffer value used to calculate the new z values.
            - rounding_deci (int): The number of decimal places to round the z
                values to.

        Returns:
            - tuple:
                - A tuple containing the intersection points of the contour
                with the objective function and a list of new z values for
                    each tile.
                - z_intersection (numpy.ndarray): The intersection points of
                    the contour with the objective function.
                - z_contour_tile_list (list): A list of new z values for each
                    tile.
        """

        z_intersection = self.objective_function([x_contour, y_contour])
        z_intersection = np.around(z_intersection, decimals=rounding_deci)

        # calculate new z values
        z_contour_tile_list = []
        for i in range(len(x_tiles)):
            z_tile = np.linspace(
                min(z_intersection) - buff,
                max(z_intersection) + buff,
                x_tiles[i].shape[0],
            )
            z_contour_tile = np.tile(z_tile, (len(z_tile), 1)).T
            z_contour_tile_list.append(z_contour_tile)
        return z_intersection, z_contour_tile_list

    @staticmethod
    def remove_dup_minima(lst):
        """Removes duplicate minima from a given list.

        Args:
            lst (list): The input list from which duplicate minima will be
                removed.

        Returns:
            list: The list with duplicate minima removed.

        This function iterates over the input list and checks for consecutive
        numbers. If the count of consecutive numbers is not 2, all consecutive
        numbers are added to the result list. If the count is 2, only the
        first of the two consecutive numbers is added to the result list. The
        function returns the result list with duplicate minima removed.

        Example:
            >>> remove_dup_minima([1, 2, 3, 3, 4, 5, 5, 6])
            [1, 2, 3, 4, 5, 6]
        """

        # Initialize an empty list to store the result
        result = []
        # Initialize a counter for consecutive numbers
        consecutive_count = 1
        # Iterate over the list using index
        for i in range(len(lst)):
            # Check if we're not at the last element and the next element is consecutive
            if i + 1 < len(lst) and lst[i] + 1 == lst[i + 1]:
                consecutive_count += 1
            else:
                # If the count is not 2, add all consecutive numbers
                if consecutive_count != 2:
                    result.extend(lst[i - consecutive_count + 1 : i + 1])
                # If the count is 2, add only the first of the two consecutive numbers
                else:
                    result.append(lst[i - 1])
                # Reset the consecutive count
                consecutive_count = 1
        return result

    @staticmethod
    def find_local_minima(numbers: list[float]) -> list[int]:
        """Finds the indices of local minima in a given list of numbers.

        Args:
            numbers (List[float]): A list of numbers.

        Returns:
            List[int]: A list of indices of local minima.

        This function iterates over the input list, skipping the first and
        last elements. It checks for a single minimum by comparing the current
        element with its neighbors. If the current element is smaller than its
        neighbors, its index is added to the list of minima indices. It also
        checks for two consecutive equal minima, and if the conditions are
        met, the index of the first minimum is added to the list.
        Additionally, it checks for three or more consecutive equal minima,
        and if the conditions are met, all indices of the consecutive equal
        minima are added to the list. The function returns the list of minima
        indices.

        Example:
            >>> find_local_minima([1, 0, 3, 3, 4, 2, 2, 6, 1, 1 ,1,2])
            [1, 5, 8, 9, 10]
        """
        # Initialize the list to store indices of minima
        minima_indices: list[int] = []

        # Iterate over the list, skipping the first and last elements
        for i in range(1, len(numbers) - 1):
            # Check for a single minimum
            if numbers[i - 1] > numbers[i] < numbers[i + 1]:
                minima_indices.append(i)
            # Check for two consecutive equal minima
            elif (
                i < len(numbers) - 2
                and numbers[i] == numbers[i + 1]
                and numbers[i - 1] > numbers[i]
                and numbers[i + 2] > numbers[i]
            ):
                minima_indices.append(i)
            # Check for three or more consecutive equal minima
            elif (
                i < len(numbers) - 3
                and numbers[i] == numbers[i + 1] == numbers[i + 2]
                and numbers[i - 1] > numbers[i]
                and numbers[i + 3] > numbers[i]
            ):
                j = i
                # Add all indices of the consecutive equal minima
                while j < len(numbers) and numbers[j] == numbers[i]:
                    minima_indices.append(j)
                    j += 1
        return minima_indices

    def solve_graphically(
        self, omit_num_res=False, z_contour_tile_buff=10, round_deci=7
    ):
        """This function solves a graphically represented optimization problem.

        Parameters:
            - omit_num_res (bool): Flag to omit numerical results in case no
            solution found by numerical methods.
            - z_contour_tile_buff (int): Buffer (Padding) for contour tiles.
            round_deci (int): Number of decimal places to round to.
        Returns:
            results_list (list): List of results including contour paths,
            intersection points, minima indices, and tile lists.
        """
        self._test_functions()
        fg0_intersection_pts, g_contour_zero_values = self._check_feasablity()
        num_opt_sol = self._numerical_solver(initial_guess=fg0_intersection_pts[0])
        if np.isnan(num_opt_sol.fun) and not omit_num_res:
            return num_opt_sol

        (
            x_contour,
            y_contour,
            x_contour_list,
            y_contour_list,
            x_contour_tile_list,
            y_contour_tile_list,
        ) = self._get_contour_paths_verticies_and_tiles(
            self.constraint_function,
            self.x_domain[0],
            self.x_domain[1],
            self.y_domain[0],
            self.y_domain[1],
            "plot",
        )

        (
            x_solv_contour,
            y_solv_contour,
            x_solv_contour_list,
            y_solv_contour_list,
            x_solv_contour_tile_list,
            y_solv_contour_tile_list,
        ) = self._get_contour_paths_verticies_and_tiles(
            self.constraint_function,
            self.x_domain[0],
            self.x_domain[1],
            self.y_domain[0],
            self.y_domain[1],
            "solv",
        )

        z_intersection, z_contour_tile_list = self._calculate_zintersec_ztiles(
            x_contour,
            y_contour,
            x_contour_tile_list,
            buff=z_contour_tile_buff,
            rounding_deci=round_deci,
        )
        z_solv_intersection, z_solv_contour_tile_list = (
            self._calculate_zintersec_ztiles(
                x_solv_contour,
                y_solv_contour,
                x_solv_contour_list,
                buff=z_contour_tile_buff,
                rounding_deci=round_deci,
            )
        )

        # Find minima
        local_minima_idx_list = []
        # Find the intersection points
        index_pad = 0
        for xc, yc in zip(x_solv_contour_list, y_solv_contour_list):
            z_intersec_temp = self.objective_function([xc, yc])
            temp_local_minima_idx_list = self.find_local_minima(z_intersec_temp)
            local_minima_idx_list.append(
                [index_pad + i for i in temp_local_minima_idx_list]
            )
            index_pad += xc.shape[0]
        flat_local_minima_idx_list = [
            item for sublist in local_minima_idx_list for item in sublist
        ]
        if len(flat_local_minima_idx_list) == 0:
            print(
                "Couldn't find minima! Solution doesn't exist. Or \
                'solv_precision_factor' is too low."
            )
            return None
        smallest_local_minima = min(
            [z_solv_intersection[i] for i in flat_local_minima_idx_list]
        )
        smallest_loc_mini_id = np.where(z_solv_intersection == smallest_local_minima)[0]
        smallest_local_minima_idx = self.remove_dup_minima(smallest_loc_mini_id)
        results_list = [
            x_solv_contour,
            y_solv_contour,
            z_solv_intersection,
            local_minima_idx_list,
            smallest_local_minima_idx,
            z_contour_tile_list,
            x_contour_tile_list,
            y_contour_tile_list,
        ]
        return results_list

    @staticmethod
    def _get_minmax_buff(buff):
        """Returns the minimum and maximum values from the given input buffer.

        Args:
            buff (Union[int, float, tuple]): The input buffer. It can be an
            integer, a float, or a tuple of length 2.

        Returns:
            tuple[Union[int, float]]: A tuple containing the minimum and
            maximum values from the input buffer.

        Raises:
            ValueError: If the input buffer is not an integer, a float, or a
            tuple of length 2.
        """
        if isinstance(buff, (int, float)):
            min_buff = max_buff = buff
        elif isinstance(buff, tuple) and len(buff) == 2:
            min_buff, max_buff = buff
        else:
            raise ValueError(
                f"{buff} must be a tuple of length 2, an int, or \
                             a float."
            )
        return min_buff, max_buff

    def visualize(
        self,
        results: Union[list, None],
        combination: str = "sol",
        z_buff: Union[int, tuple] = 3,
        x_buff: Union[int, tuple] = 3,
        y_buff: Union[int, tuple] = 3,
    ):
        """This method takes in a list of results and a combination string as
        parameters, and returns a Plotly figure and a dictionary of figures.

        The method performs the following steps:

        1. Sets up buffer values for the x, y, and z axes based on the input parameters.
        2. Checks if the results parameter is a list. If it is, it extracts several variables from the list. If results is None, it sets a flag `plot_init` to True.
        3. Splits the combination string into a list of elements and removes any duplicate elements.
        4. Initializes an empty dictionary `figs` to store the figures.
        5. If the combination string contains 'f', it creates a surface plot for the objective function using the `_get_plotly_fig` method and stores it in `figs` with the key 'f_surface'.
        6. If the combination string contains 'g', it creates a surface plot for the constraint function using the `_get_plotly_fig` method and stores it in `figs` with the key 'g_surface'.
        7. If the combination string contains 'sol' or 'sol-g0', it calculates the neighborhood of the smallest local minimum, creates surface plots for the objective function in the neighborhood and the intersection points, and stores them in `figs` with the keys 'f_surface_minima', 'intersection_scatter', and 'local_minima_scatter'. It also stores the global minimum as a scatter plot in `figs` with the key 'minima_scatter'.
        8. If the combination string contains 'loc', 'sol', or 'sol-g0', it creates a scatter plot for the global minimum and stores it in `figs` with the key 'minima_scatter'.
        9. If the combination string contains 'g0', it creates surface plots for the extrusion of the feasible region along the z-axis and stores them in `figs` with the key 'contour0_extrusion'.
        10. Creates a title for the plot based on the combination string and the keys in `figs`.
        11. Creates a Plotly figure by combining the figures stored in `figs` based on the combination string.
        12. Updates the layout of the figure and returns it along with `figs`.

        Args:
            - results: The restults returned by the `solve_graphycally` method.
            - combination: The combination of the plots to be created. \
                It can contain'f', 'g', 'sol', 'loc', 'sol-g0', or 'g0'. \
                the combination has to be with '+' as separators.
            - z_buff: The buffer for the z-axis.
            - x_buff: The buffer for the x-axis.
            - y_buff: The buffer for the y-axis.

        Returns:
            tuple: A tuple of the Plotly figure and a dictionary of figures.
        """

        xmin_buff, xmax_buff = self._get_minmax_buff(x_buff)
        ymin_buff, ymax_buff = self._get_minmax_buff(y_buff)
        zmin_buff, zmax_buff = self._get_minmax_buff(z_buff)

        if isinstance(results, list):
            (
                x_contour,
                y_contour,
                z_intersection,
                local_minima_idx_list,
                smallest_local_minima_idx,
                z_contour_tile_list,
                x_contour_tile_list,
                y_contour_tile_list,
            ) = results
        if results is None:
            plot_init = True
        else:
            plot_init = False
        combination_l = combination.split("+")
        combination_l = list(set(combination_l))

        figs = {}
        if ("f" in combination_l) or plot_init:
            ## plot fig for f and g with its contour separately on the user defined domain
            f_plot_kwargs = {
                "colorscale": "Plotly3",
                "name": "Objective<br>function",
                "opacity": 1,
                "colorbar": dict(len=0.5, y=0.65),
                "showlegend": True,
            }
            f_plot_args = [
                self.objective_function,
                self.x_domain[0],
                self.x_domain[1],
                self.y_domain[0],
                self.y_domain[1],
            ]
            f_surface, z_f = self._get_plotly_fig(
                "surface", f_plot_args, **f_plot_kwargs
            )
            figs["f_surface"] = (f_surface, z_f)

        if ("g" in combination_l) or plot_init:
            g_plot_kwargs = {
                "colorscale": "greys",
                "name": "Constraint<br>function",
                "opacity": 1,
                "showlegend": True,
            }
            g_plot_args = [
                self.constraint_function,
                self.x_domain[0],
                self.x_domain[1],
                self.y_domain[0],
                self.y_domain[1],
            ]
            g_surface, z_g = self._get_plotly_fig(
                "surface", g_plot_args, **g_plot_kwargs
            )
            figs["g_surface"] = (g_surface, z_g)

        if ("sol" in combination_l) or ("sol-g0" in combination_l) and not plot_init:
            minx = min(x_contour[smallest_local_minima_idx]) - xmin_buff
            miny = min(y_contour[smallest_local_minima_idx]) - ymin_buff
            maxx = max(x_contour[smallest_local_minima_idx]) + xmax_buff
            maxy = max(y_contour[smallest_local_minima_idx]) + ymax_buff
            sol_neighborhood = {"xmin": minx, "xmax": maxx, "ymin": miny, "ymax": maxy}
            # plot f in minima neighborhood
            f_minima_plot_kwargs = {
                "colorscale": "Plotly3",
                "name": "Objective<br>function",
                "opacity": 1,
                "colorbar": dict(len=0.5, y=0.65),
                "showlegend": True,
            }
            f_surface_minima_args = [self.objective_function, minx, maxx, miny, maxy]
            f_surface_minima, z_f_minima = self._get_plotly_fig(
                "surface", f_surface_minima_args, **f_minima_plot_kwargs
            )
            figs["f_surface_minima"] = (f_surface_minima, z_f_minima)

            # intersection_scatter_kwargs = {"mode":'lines',
            #                        "line":dict(color='red',width=7,dash='solid'),
            #                        "name":'Intersection<br>Points'}
            intersection_scatter_kwargs = {
                "mode": "markers",
                "marker": dict(color="red", size=5, opacity=0.8),
                "name": "Intersection<br>Points",
            }
            intersection_scatter_args = [x_contour, y_contour, z_intersection]
            intersection_scatter = self._get_plotly_fig(
                "scatter", intersection_scatter_args, **intersection_scatter_kwargs
            )
            figs["intersection_scatter"] = (intersection_scatter, z_intersection)

            x_local_minima_list = []
            y_local_minima_list = []
            z_local_minima_list = []
            for l in local_minima_idx_list:
                x_local_minima_list.extend([x_contour[i] for i in l])
                y_local_minima_list.extend([y_contour[i] for i in l])
                z_local_minima_list.extend(
                    [self.objective_function([x_contour[i], y_contour[i]]) for i in l]
                )
            local_minima_scatter_kwargs = {
                "mode": "markers",
                "marker": dict(symbol="cross", color="yellow", size=13),
                "name": "Local<br>minima",
            }
            local_minima_scatter_args = [
                x_local_minima_list,
                y_local_minima_list,
                z_local_minima_list,
            ]
            local_minima_scatter = self._get_plotly_fig(
                "scatter", local_minima_scatter_args, **local_minima_scatter_kwargs
            )
            figs["local_minima_scatter"] = (local_minima_scatter, z_local_minima_list)

        if (
            "loc" in combination_l
            or "sol" in combination_l
            or "sol-g0" in combination_l
        ) and not plot_init:
            minima_scatter_kwargs = {
                "mode": "markers",
                "marker": dict(symbol="x", color="green"),
                "name": "Global<br>minimum",
            }
            x_minima = x_contour[smallest_local_minima_idx]
            y_minima = y_contour[smallest_local_minima_idx]
            z_minima = self.objective_function(
                [
                    x_contour[smallest_local_minima_idx],
                    y_contour[smallest_local_minima_idx],
                ]
            )
            minima_scatter_args = [x_minima, y_minima, z_minima]
            minima_scatter = self._get_plotly_fig(
                "scatter", minima_scatter_args, **minima_scatter_kwargs
            )
            figs["minima_scatter"] = (minima_scatter, z_minima)

        if ("g0" in combination_l or "sol" in combination_l) and not plot_init:
            # Create the surface plot
            figs["contour0_extrusion"] = ([], z_intersection)
            for i in range(len(z_contour_tile_list)):
                contour0_extrusion_kwargs = {
                    "colorscale": "Greys",
                    "opacity": 0.7,
                    "name": "Equality<br>Constraint",
                    "colorbar": dict(len=0.5, y=0.1),
                    "legendgroup": "g0group",
                }
                if i == 0:
                    contour0_extrusion_kwargs["showlegend"] = True
                contour0_extrusion_args = [
                    x_contour_tile_list[i],
                    y_contour_tile_list[i],
                    z_contour_tile_list[i],
                ]
                fig = self._get_plotly_fig(
                    "surface", contour0_extrusion_args, **contour0_extrusion_kwargs
                )
                figs["contour0_extrusion"][0].append(fig)

        title_dic = {
            "f": "Objective function",
            "g": "Constraint function",
            "g0": "Extrusion of the feasable region along z-axis",
            "loc": "Local view of the ",
            "sol": "Local view of the optimal solution",
            "sol-g0": "Local view of the optimal solution without constraint \
            extrusion",
        }
        title_comb = [elem for elem in title_dic.keys() if elem in combination_l]
        if "loc" in title_comb:
            title = title_dic["loc"]
            title_comb.remove("loc")
            title += ", ".join([title_dic[i] for i in title_comb])
        elif "sol" in title_comb:
            title = title_dic["sol"]
        else:
            title = "Plot of the " + ", ".join([title_dic[i] for i in title_comb])

        fig_data = []
        z_list = []
        for i in combination_l:
            if i == "sol" and not plot_init:
                fig_data = [
                    figs["minima_scatter"][0],
                    figs["intersection_scatter"][0],
                    figs["local_minima_scatter"][0],
                    figs["f_surface_minima"][0],
                    *figs["contour0_extrusion"][0],
                ]
                z_list = [figs["minima_scatter"][1]]
                x_axis = dict(
                    range=[sol_neighborhood["xmin"], sol_neighborhood["xmax"]]
                )
                y_axis = dict(
                    range=[sol_neighborhood["ymin"], sol_neighborhood["ymax"]]
                )
                break
            if i == "sol-g0" and not plot_init:
                fig_data = [
                    figs["minima_scatter"][0],
                    figs["intersection_scatter"][0],
                    figs["local_minima_scatter"][0],
                    figs["f_surface_minima"][0],
                ]
                z_list = [figs["minima_scatter"][1]]
                x_axis = dict(
                    range=[sol_neighborhood["xmin"], sol_neighborhood["xmax"]]
                )
                y_axis = dict(
                    range=[sol_neighborhood["ymin"], sol_neighborhood["ymax"]]
                )
                break
            if i == "f" or plot_init:
                fig_data.append(figs["f_surface"][0])
                z_list.append(figs["f_surface"][1])
            if i == "g":
                fig_data.append(figs["g_surface"][0])
                z_list.append(figs["g_surface"][1])
            if i == "g0" and not plot_init:
                fig_data.extend(figs["contour0_extrusion"][0])
                z_list.append(figs["contour0_extrusion"][1])
        if "loc" in combination_l and not plot_init:
            z_list = [figs["minima_scatter"][1]]
        maxz_list = [np.nanmax(arr) for arr in z_list]
        minz_list = [np.nanmin(arr) for arr in z_list]
        minz = max(minz_list) - zmin_buff
        maxz = min(maxz_list) + zmax_buff

        # Create the surface plot
        fig = go.Figure(data=fig_data)
        # Update the layout of the plot
        if i == "sol" or i == "sol-g0" and not plot_init:
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    xaxis=x_axis,
                    yaxis=y_axis,
                    zaxis=dict(range=[minz, maxz]),
                ),
                width=1000,
                height=700,
                hovermode="x unified",
            )
            fig.data[3].cmin = max(minz, np.nanmin(figs["f_surface_minima"][1]))
            fig.data[3].cmax = min(maxz, np.nanmax(figs["f_surface_minima"][1]))
            if len(fig.data) > 4:
                for i in range(4, len(fig.data)):
                    fig.data[i].cmin = minz
                    fig.data[i].cmax = maxz
        else:
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    zaxis=dict(range=[minz, maxz]),
                ),
                width=500,
                height=500,
                hovermode="x unified",
            )
            if len(fig_data) == 1:
                fig.update_traces(colorbar=dict(len=0.8, y=0.5))
            fig.update_traces(cmin=minz, cmax=maxz)
            fig.update_traces(
                contours_z=dict(
                    start=0,
                    end=1,
                    size=1,
                    show=True,
                    usecolormap=False,
                    highlightcolor="red",
                    project_z=True,
                ),
                selector=dict(type="surface", name="Constraint<br>function"),
            )
            fig.update_traces(showscale=False, selector=dict(type="surface"))
        fig.update_layout(
            legend=dict(
                yanchor="bottom",
                xanchor="left",
                x=0.2,
                y=-0.1,
                itemsizing="constant",
                orientation="h",
            )
        )

        # Show the plot
        return fig, figs

    @staticmethod
    def numerical_gradient(func, x0, y0, h=1e-5):
        """Calculate the numerical gradient of a 2D function at a given point
        (x0, y0).

        Parameters:
            - func (function): The 2D function to calculate the gradient for.
            - x0 (float): The x-coordinate of the point.
            - y0 (float): The y-coordinate of the point.
            - h (float): The step size for numerical differentiation (default
            is 1e-5).

        Returns:
            tuple: A tuple containing the numerical gradients df/dx and df/dy.
        """
        df_dx = (func((x0 + h, y0)) - func((x0, y0))) / h
        df_dy = (func((x0, y0 + h)) - func((x0, y0))) / h
        return df_dx, df_dy

    @staticmethod
    def find_closest_point(points, target_point):
        """Find the closest point to a given target point from a set of points.

        Parameters:
            - points (np.ndarray): An array of shape (n, m) where n is the
            number of points and m is the dimensionality of each point.
            - target_point (np.ndarray): A 1D array of length m representing
            the target point.

        Returns:
            tuple: A tuple containing the closest point from the points array
            to the target point and its index in the points array.
        """
        # Calculate the Euclidean distance between each point and the target point
        distances = np.linalg.norm(points - target_point, axis=1)

        # Find the index of the minimum distance
        closest_index = np.argmin(distances)

        # Retrieve the closest point
        closest_point = points[closest_index]

        return closest_point, closest_index

    def solve_PGD(
        self,
        progress_bar,
        init_pt: Union[str, tuple, list] = "auto",
        max_iter=1000,
        alpha=0.1,
        tol=1e-6,
    ):
        """Solves the Projected Gradient Descent (PGD) algorithm to find the
        minimum of a function subject to a constraint.

        Parameters:
            - progress_bar (ProgressBar): A progress bar object to display the
            progress of the algorithm.
            - init_pt (tuple or str, optional): The initial point for the
            algorithm. If 'auto', the initial point is set to the first
            intersection point of the constraint function with the x-axis.
            Default is 'auto'.
            - max_iter (int, optional): The maximum number of iterations for
            the algorithm. Default is 1000.
            - alpha (float, optional): The step size for the gradient descent
            step. Default is 0.1.
            - tol (float, optional): The tolerance for convergence. If the norm
            of the difference between consecutive points is less than tol, the
            algorithm is considered converged. Default is 1e-6.

        Returns:
            list: A list of tuples representing the coordinates of the points
            visited during the algorithm.
        """
        steps = []
        self._test_functions()
        fg0_intersection_pts, g_contour_zero_values = self._check_feasablity()
        x_solv_contour, y_solv_contour, _, _, _, _ = (
            self._get_contour_paths_verticies_and_tiles(
                self.constraint_function,
                self.x_domain[0],
                self.x_domain[1],
                self.y_domain[0],
                self.y_domain[1],
                "solv",
            )
        )
        feasable_points = np.vstack([x_solv_contour, y_solv_contour]).T
        if init_pt == "auto":
            x, y = fg0_intersection_pts[0]
            prev_x, prev_y = fg0_intersection_pts[0]
        else:
            x, y = init_pt
            prev_x, prev_y = init_pt
        steps.append((x, y))

        for i in range(max_iter):
            progress_bar.progress(50 * i // max_iter, text="Running PGD..")
            # Compute gradient of the objective function
            grad_f = self.numerical_gradient(self.objective_function, x, y)
            # Gradient descent step
            x = x - alpha * grad_f[0]
            y = y - alpha * grad_f[1]
            steps.append((x, y))

            # Projection onto the constraint g(x, y) = 0
            closest_point, closest_index = self.find_closest_point(
                feasable_points, (x, y)
            )
            x = closest_point[0]
            y = closest_point[1]
            steps.append((x, y))

            # Check for convergence
            if np.linalg.norm((x - prev_x, y - prev_y)) < tol:
                progress_bar.progress(50, text="Running PGD..")
                break
            else:
                prev_x = x
                prev_y = y
        return steps

    def visualize_PGD(
        self,
        pgd_steps: list,
        z_buff: Union[int, tuple] = 3,
        x_buff: Union[int, tuple] = 3,
        y_buff: Union[int, tuple] = 3,
        grad_field_scale: float = 0.2,
    ):
        """Visualize the projected gradient descent (PGD) steps.

        Args:
            - pgd_steps (list): A list of PGD steps.
            - z_buff (Union[int, tuple], optional): The buffer for the z-axis.
            Defaults to 3.
            - x_buff (Union[int, tuple], optional): The buffer for the x-axis.
            Defaults to 3.
            - y_buff (Union[int, tuple], optional): The buffer for the y-axis.
            Defaults to 3.
            - grad_field_scale (float, optional): The scale of the gradient
            field. Defaults to 0.2.

        Returns:
            go.Figure: The plotly figure containing the visualization.

        Example usage:

        ```
        pgd_steps = [...]
        fig = visualize_PGD(pgd_steps)
        fig.show()
        ```

        Note:
        - The `pgd_steps` list should contain tuples or lists of x and y
        coordinates.
        - The `z_buff`, `x_buff`, and `y_buff` arguments can be either
        integers or tuples of two integers.
        - The `grad_field_scale` argument determines the scale of the
        gradient field.
        """

        # scatter PGD steps data
        pgd_steps_arr = np.vstack(pgd_steps)
        scatter_x = pgd_steps_arr[:, 0]
        scatter_y = pgd_steps_arr[:, 1]
        scatter_z = self.objective_function((scatter_x, scatter_y))
        xsol = scatter_x[-1]
        ysol = scatter_y[-1]

        xmin_buff, xmax_buff = self._get_minmax_buff(x_buff)
        ymin_buff, ymax_buff = self._get_minmax_buff(y_buff)
        zmin_buff, zmax_buff = self._get_minmax_buff(z_buff)

        minx = max(min(scatter_x) - xmin_buff, self.x_domain[0])
        miny = max(min(scatter_y) - ymin_buff, self.y_domain[0])
        minz = min(scatter_z) - zmin_buff
        maxx = min(max(scatter_x) + xmax_buff, self.x_domain[1])
        maxy = min(max(scatter_y) + ymax_buff, self.y_domain[1])
        maxz = max(scatter_z) + zmax_buff

        # scatter PGD steps data
        x = np.linspace(minx, maxx, self.x_precision)
        y = np.linspace(miny, maxy, self.y_precision)
        x_grid, y_grid = np.meshgrid(x, y)
        zf = self.objective_function((x_grid, y_grid))
        zg = self.constraint_function((x_grid, y_grid))

        # scatter PGD steps data
        x_grad_field = np.linspace(minx, maxx, self.x_precision // 2)
        y_grad_field = np.linspace(miny, maxy, self.y_precision // 2)
        # x_grad_field_focused = self.smooth_focus_linspace(x_grad_field, xsol)
        # y_grad_field_focused = self.smooth_focus_linspace(y_grad_field, ysol)

        x_grad_field_grid, y_grad_field_grid = np.meshgrid(x_grad_field, y_grad_field)

        zf_grad_field = self.objective_function((x_grad_field_grid, y_grad_field_grid))
        v, u = np.gradient(zf_grad_field, 0.5, 0.5)
        grad_sol_f = self.numerical_gradient(
            self.objective_function, xsol, ysol, h=1e-7
        )
        grad_sol_g = self.numerical_gradient(
            self.constraint_function, xsol, ysol, h=1e-7
        )
        print(grad_sol_f, grad_sol_g)

        x_plot_contour, y_plot_contour, _, _, _, _ = (
            self._get_contour_paths_verticies_and_tiles(
                self.constraint_function, minx, maxx, miny, maxy, "plot"
            )
        )
        z_intersection = self.objective_function([x_plot_contour, y_plot_contour])

        # plot fig for f and g with its contour separately on the user defined domain
        f_plot_kwargs = {
            "colorscale": "Plotly3",
            "name": "Objective<br>function",
            "opacity": 1,
            "colorbar": dict(len=0.8, y=0.35),
            "showlegend": True,
        }
        f_plot_args = [self.objective_function, minx, maxx, miny, maxy]
        f_surface, z_f = self._get_plotly_fig("surface", f_plot_args, **f_plot_kwargs)

        intersection_scatter_kwargs = {
            "mode": "markers",
            "marker": dict(color="red", size=3, opacity=0.5, symbol="circle"),
            "name": "Intersection<br>Points",
        }
        intersection_scatter_args = [x_plot_contour, y_plot_contour, z_intersection]
        intersection_scatter = self._get_plotly_fig(
            "scatter", intersection_scatter_args, **intersection_scatter_kwargs
        )

        contour_g = go.Contour(
            z=zg,
            x=x,
            y=y,
            colorscale="Burg",  # Colorscale for the contours
            contours_coloring="lines",
            contours=dict(
                start=0,  # Start level for contours
                end=0,  # End level for contours
                showlabels=True,  # Show labels on contours
                labelfont=dict(
                    size=8,
                    color="black",
                ),
            ),
            line=dict(
                color="black",  # Color of contour lines
                width=2,  # Width of contour lines
                dash="solid",  # style of contour lines [dash, solid, dot, dashdot, longdash, longdashdot]
            ),
            showlegend=False,
        )

        contour_f = go.Contour(
            z=zf,
            x=x,
            y=y,
            colorscale="Plotly3",  # Colorscale for the contours
            contours_coloring="lines",
            contours=dict(
                start=maxz,  # Start level for contours
                end=minz,  # End level for contours
                showlabels=True,  # Show labels on contours
                labelfont=dict(
                    size=8,
                    color="black",
                ),
            ),
            ncontours=20,  # Number of contour lines
            line=dict(
                color="black",  # Color of contour lines
                width=2,  # Width of contour lines
                dash="solid",  # style of contour lines [dash, solid, dot, dashdot, longdash, longdashdot]
            ),
            showlegend=False,
        )

        # Create quiver figure
        fig_grad_field = ff.create_quiver(
            x_grad_field_grid,
            y_grad_field_grid,
            u,
            v,
            scale=grad_field_scale,
            arrow_scale=grad_field_scale,
            name="Gradient field",
            line_width=1,
        )
        for trace in fig_grad_field.data:
            trace.line.color = "gray"

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "contour"}, {"type": "scatter"}],
                [{"colspan": 2, "type": "scene"}, None],
            ],
            vertical_spacing=0.08,  # Adjust vertical spacing between rows
            horizontal_spacing=0.08,  # Adjust horizontal spacing between columns
            subplot_titles=("Contour", "Gradient field", "Surface"),
        )

        fig.add_trace(contour_f, row=1, col=1)
        fig.add_trace(contour_g, row=1, col=1)

        fig.add_traces(data=fig_grad_field.data, rows=1, cols=2)
        fig.add_trace(contour_g, row=1, col=2)

        fig.add_trace(f_surface, row=2, col=1)
        fig.add_trace(intersection_scatter, row=2, col=1)

        sizes = np.array(list(range(scatter_z.shape[0])))
        sizes = list(5 * sizes / max(sizes) + 3)
        sizes.reverse()

        # Add traces, one for each slider step
        for i in range(len(scatter_x)):
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    mode="lines+markers",
                    marker=dict(
                        symbol="x",
                        size=sizes,
                        color=scatter_z,  # set color to an array/list of desired values
                        colorscale="Burg",  # choose a colorscale
                    ),
                    line=dict(
                        color="black",  # Line color
                        # colorscale='Viridis',
                        width=1,  # Line width
                    ),
                    opacity=0.8,
                    name=f"Step {i+1}",
                    x=scatter_x[: i + 1],
                    y=scatter_y[: i + 1],
                ),
                row=1,
                col=1,
            )

        # Add traces, one for each slider step
        for i in range(len(scatter_x)):
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    mode="lines+markers",
                    marker=dict(
                        symbol="circle",
                        size=sizes,
                        color=scatter_z,  # set color to an array/list of desired values
                        colorscale="Burg",  # choose a colorscale
                    ),
                    line=dict(
                        color="black",  # Line color
                        # colorscale='Viridis',
                        width=1,  # Line width
                    ),
                    opacity=0.8,
                    name=f"Step {i+1}",
                    x=scatter_x[: i + 1],
                    y=scatter_y[: i + 1],
                ),
                row=1,
                col=2,
            )

        # Add traces, one for each slider step
        for i in range(len(scatter_x)):
            fig.add_trace(
                go.Scatter3d(
                    visible=False,
                    mode="lines+markers",
                    marker=dict(
                        symbol="diamond",
                        size=sizes,
                        color=scatter_z,  # set color to an array/list of desired values
                        colorscale="Burg",  # choose a colorscale
                        opacity=1,
                    ),
                    line=dict(
                        color=scatter_z,  # Line color
                        colorscale="Burg",
                        width=2,  # Line width
                    ),
                    name=f"Step {i+1}",
                    x=scatter_x[: i + 1],
                    y=scatter_y[: i + 1],
                    z=scatter_z[: i + 1],
                ),
                row=2,
                col=1,
            )

        # Make the first scatter trace visible
        fig.data[len(fig.data) - len(scatter_x)].visible = True
        fig.data[len(fig.data) - 2 * len(scatter_x)].visible = True
        fig.data[len(fig.data) - 3 * len(scatter_x)].visible = True

        # Create and add slider
        steps = []
        for i in range(len(scatter_x)):
            step = dict(
                method="update",
                args=[
                    {"visible": [False] * len(fig.data)},
                    {"title": f"Projected Gradient Descent Step {i+1}"},
                ],  # layout attribute
            )
            # Toggle visibility of the surface and i'th scatter trace
            for j in range(6):
                step["args"][0]["visible"][j] = True  # Keep surface plot visible
            step["args"][0]["visible"][len(fig.data) - 3 * len(scatter_x) + i] = True
            step["args"][0]["visible"][len(fig.data) - 2 * len(scatter_x) + i] = True
            step["args"][0]["visible"][len(fig.data) - len(scatter_x) + i] = True
            steps.append(step)

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "PGD Step: "},
                pad={"t": 10},
                steps=steps,
            )
        ]

        fig.add_annotation(
            x=grad_sol_f[0] / (np.linalg.norm(grad_sol_f)) + scatter_x[-1],
            y=grad_sol_f[1] / (np.linalg.norm(grad_sol_f)) + scatter_y[-1],
            ax=scatter_x[-1],
            ay=scatter_y[-1],
            xref="x2",
            yref="y2",
            axref="x2",
            ayref="y2",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="blue",
        )
        fig.add_annotation(
            x=1.5 * grad_sol_f[0] / (np.linalg.norm(grad_sol_f))
            + scatter_x[-1],  # Text x position with offset
            y=1.5 * grad_sol_f[1] / (np.linalg.norm(grad_sol_f))
            + scatter_y[-1],  # Text y position with offset
            text="∇f(x*)",  # Annotation text
            showarrow=False,  # No arrow for this annotation
            xref="x2",  # x reference is the data coordinates
            yref="y2",  # y reference is the data coordinates
            font=dict(size=15, color="blue"),  # Text font size  # Text color
            align="center",  # Text alignment
        )

        fig.add_annotation(
            x=grad_sol_g[0] / (np.linalg.norm(grad_sol_g)) + scatter_x[-1],
            y=grad_sol_g[1] / (np.linalg.norm(grad_sol_g)) + scatter_y[-1],
            ax=scatter_x[-1],
            ay=scatter_y[-1],
            xref="x2",
            yref="y2",
            axref="x2",
            ayref="y2",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
        )
        fig.add_annotation(
            x=1.5 * grad_sol_g[0] / (np.linalg.norm(grad_sol_g))
            + scatter_x[-1],  # Text x position with offset
            y=1.5 * grad_sol_g[1] / (np.linalg.norm(grad_sol_g))
            + scatter_y[-1],  # Text y position with offset
            text="∇g(x*)",  # Annotation text
            showarrow=False,  # No arrow for this annotation
            xref="x2",  # x reference is the data coordinates
            yref="y2",  # y reference is the data coordinates
            font=dict(size=15, color="red"),  # Text font size  # Text color
            align="center",  # Text alignment
        )

        fig.update_layout(
            title="Projected gradient descent",
            sliders=sliders,
            scene1=dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis",
                xaxis=dict(range=[minx, maxx]),
                yaxis=dict(range=[miny, maxy]),
                zaxis=dict(range=[minz, maxz]),
            ),
            width=1000,
            height=1000,
        )
        fig.update_xaxes(
            range=[minx, maxx], row=1, col=1
        )  # Update x-axis range for contour plot
        fig.update_xaxes(
            range=[minx, maxx], row=1, col=2
        )  # Update x-axis range for scatter plot
        fig.update_yaxes(
            range=[miny, maxy], row=1, col=1
        )  # Update y-axis range for contour plot
        fig.update_yaxes(
            range=[miny, maxy], row=1, col=2
        )  # Update y-axis range for scatter plot
        fig.data[4].cmin = minz
        fig.data[4].cmax = maxz

        fig.update_traces(showscale=False, selector=dict(type="contour"))

        return fig


class TexPixToPython:
    def __init__(self, gemenai_model) -> None:
        """Initializes the instance of the class.

        Args:
            gemenai_model (Any): The gemenai model.

        Returns:
            None
        """
        self.gemenai_model = gemenai_model
        self.genai_generation_config = genai.types.GenerationConfig(
            candidate_count=1, max_output_tokens=200, temperature=0.1
        )

    def convert_tex_to_python(self, img_path):
        """Converts a LaTeX optimization problem image to a Python dictionary
        of extracted functions.

        Args:
            img_path (str): The path to the image file of the optimization
            problem.

        Returns:
            dict: A dictionary containing the extracted objective function and
            equality constraint as Python code.
                The keys of the dictionary are "obj_func" and "constraint_func".
                The values are the corresponding function definitions as
                strings.
        """

        tex_img = Image.open(img_path)
        prompt = f"""From the following optimization problem, extract the \
            objective function and the equality constraint. Your response \
            should be formatted like this:

            def objective_func(x,y):
                return f(x,y)

            def constraint_func(x,y):
                return g(x,y)

            The constraint must be in the format g(x,y)=0, You must return \
            both function in Python code."""
        response = self.gemenai_model.generate_content([prompt, tex_img])
        extracted_functions = [
            f.split("return")[1].strip(" ")
            for f in response.text.strip("```")[6:].strip("\n").split("\n\n")
        ]
        extracted_functions_dic = {
            "obj_func": extracted_functions[0],
            "constraint_func": extracted_functions[1],
        }
        return extracted_functions_dic


class LLMSolver:
    def __init__(self, gemenai_model) -> None:
        """Initializes a new instance of the class.

        Args:
            gemenai_model (object): An instance of the Gemenai model.

        Returns:
            None

        Initializes the following instance variables:
            - gemenai_model (object): The Gemenai model instance.
            - genai_generation_config (genai.types.GenerationConfig):
            The generation configuration for the Gemenai model.
            - safety_settings (dict): A dictionary containing safety
            settings for the Gemenai model.

        The safety_settings dictionary has the following keys and values:
            - HarmCategory.HARM_CATEGORY_HATE_SPEECH (HarmCategory):
            The block threshold for hate speech.
            - HarmCategory.HARM_CATEGORY_HARASSMENT (HarmCategory):
            The block threshold for harassment.
        """
        self.gemenai_model = gemenai_model
        self.genai_generation_config = genai.types.GenerationConfig(
            candidate_count=1, max_output_tokens=5000, temperature=1.0
        )
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def generate_report(self, obj_function: str, constraint_function: str, prog_bar):
        """A function that generates a report based on the given objective
        function and constraint function.

        It provides step-by-step instructions for solving an
        optimization problem in 2 dimensions with one equality
        constraint. The function communicates with the user and model to
        guide them through the problem-solving process. It includes
        messages, instructions, and progress updates. Returns a list of
        responses generated during the process.
        """
        messages = [
            {
                "role": "user",
                "parts": [
                    """You are a genius mathematician specialized in solving \
                        optimization problems and systems of equations.
                        You will be given an optimization problem in 2 \
                            dimension with one equality constraint.
                        The problem will be given on the following form, \
                            example:
                        "Objective function: f(x,y)= x**2+y
                        Constraint: g(x,y)=exp(x-y)".
                    I will guide you thrhought the process and you need to \
                    follow the steps I tell you. When you answer don't \
                    re-state my instructions.
                    Remember to keep your answers clear and use clear regular \
                    html mathematical notations. In case you need to use \
                    unfamiliar notation, make sure to explain it briefly."""
                    + f"Optimization problem:\nObjective function: \
                    {obj_function}\nConstraint: {constraint_function}"
                ],
            },
            {
                "role": "model",
                "parts": [
                    """I understand this task and I am waiting for your \
                        instructions. I will keep my answers clear and consice \
                        and follow the steps you provide."""
                ],
            },
        ]
        steps_instructions = [
            "Check that the feasible region isn't empty.",
            "Apply first order necessary condition to find critical points \
                (Write the Lagrangian form of the original problem and solve \
                equations system). You must show how to solve the equations \
                system to find the critical points with their respective \
                lagrange multiplier value. You must take time and verify \
                the system solutions, every thing depends on this!",
            "Check critical points regularity (constraint qualification) by \
                evaluating the gradient of the constraint at those points, if \
                it is null the point is then (non regular) else it is \
                (regular).",
            "Calculate the Hessian matrix of the Lagrangian with respect to x \
                and y, take time and verify your calculations. Give the \
                general form of the matrix with x and y. Don't evaluate it in \
                the critical points yet.",
            "Check the necessary condition of the second order in each \
                critical point. evalueate the hessian in each critical point \
                with their respective lagrange multiplier value, then \
                calculate the determinant and trace of the hessian at that \
                point and depending on their sign conclude whether the hessian \
                is positive semi-definite, or negative semi-definite or \
                indefinite. the determinant represent the product of the \
                eigenvalues and the trace represent the sum. In this step do \
                not conclude whether the points are solutions of the problem \
                or not.",
            ", in case the necessary condition was inconclusive, Check the \
                sufficient condition of the second order for each candidate \
                (meaning that a stationary point becomes a solution if the \
                hessian matrix when projected onto the feasible direction and \
                evaluated in that stationary point is found positive definite)\
                .",
            "Write a short conclusion about this optimization problem, and \
                indicate if we found solutions or not and whether they are \
                local or global.",
        ]
        for i, step_inst in tqdm(enumerate(steps_instructions)):
            prompt = f"The step number {i+1}: you need to {step_inst}. stick \
                to the task and don't go beyond that, and use markdown \
                formatting for your response."
            messages.append({"role": "user", "parts": prompt})
            response = self.gemenai_model.generate_content(
                messages,
                generation_config=self.genai_generation_config,
                safety_settings=self.safety_settings,
            )
            messages.append({"role": "model", "parts": [response.text]})
            prog_bar.progress(
                50 + (i + 1) * 50 // 7,
                text="Generating \
                              report..",
            )
            w = np.random.random() * 5 + 25
            time.sleep(w)
        response_list = [dic["parts"][0] for dic in messages if dic["role"] == "model"][
            1:
        ]
        return response_list
