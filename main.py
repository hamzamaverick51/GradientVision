# gradient_field_analysis.py

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from scipy.stats import entropy

class Gradient3DAnalyzer:
    """
    Analyzes gradient fields for 2D and 3D functions using symbolic computation.
    """
    def __init__(self):
        self.vars = sp.symbols('x y z')
        
    def compute_gradient(self, f_expr):
        """Compute the gradient vector of a symbolic expression."""
        # Determine dimension based on free symbols
        dim = len(f_expr.free_symbols)
        return sp.Matrix([sp.diff(f_expr, var) for var in self.vars[:dim]])
    
    def hessian(self, f_expr):
        """Compute the Hessian matrix of a symbolic expression."""
        dim = len(f_expr.free_symbols)
        return sp.hessian(f_expr, self.vars[:dim])
    
    def find_critical_points(self, f_expr, method='hybrid'):
        """Find critical points of a function using various methods."""
        dim = len(f_expr.free_symbols)
        grad = self.compute_gradient(f_expr)
        
        # Try symbolic solution first
        if method in ('symbolic', 'hybrid'):
            try:
                solutions = sp.solve(grad, self.vars[:dim], dict=True)
                verified = []
                
                for sol in solutions:
                    try:
                        # Convert symbolic solution to float values
                        pt = tuple(float(sol.get(var, 0)) for var in self.vars[:dim])
                        
                        # Verify if it's actually a critical point
                        if VerificationSuite.numerical_gradient_check(f_expr, pt):
                            verified.append(pt)
                    except (TypeError, ValueError):
                        # Skip points with complex values or other issues
                        continue
                
                if verified:
                    return verified
                    
            except Exception:
                pass
        
        # Fall back to numerical methods
        return self._adaptive_grid_search(f_expr)
    
    def _adaptive_grid_search(self, f_expr, ranges=((-5, 5), (-5, 5), (-5, 5))):
        """Search for critical points by evaluating gradient on progressively finer grids."""
        dim = len(f_expr.free_symbols)
        grad = self.compute_gradient(f_expr)
        points = []
        
        # Use increasingly fine resolutions to search for critical points
        for resolution in [1.0, 0.5, 0.2]:
            grids = [np.arange(r[0], r[1], resolution) for r in ranges[:dim]]
            meshes = np.meshgrid(*grids)
            
            for coords in zip(*[m.ravel() for m in meshes]):
                pt = tuple(round(c, 2) for c in coords)
                try:
                    # Substitute coordinates into gradient and check if close to zero
                    subs = {var: val for var, val in zip(self.vars[:dim], pt)}
                    if all(abs(g.subs(subs).evalf()) < 1e-5 for g in grad):
                        points.append(pt)
                except:
                    continue
        
        # Remove duplicates and return
        unique_points = []
        for pt in points:
            if not any(np.allclose(pt, up, rtol=1e-3) for up in unique_points):
                unique_points.append(pt)
                
        return unique_points
    
    def classify_critical_point(self, f_expr, point):
        """
        Classify a critical point as minimum, maximum, or saddle point.
        
        Args:
            f_expr: SymPy expression representing the function
            point: Tuple representing the critical point coordinates
            
        Returns:
            String: 'minimum', 'maximum', 'saddle', or 'unknown'
        """
        dim = len(f_expr.free_symbols)
        H = self.hessian(f_expr)
        
        try:
            # Evaluate Hessian at the critical point
            H_eval = H.subs({var: val for var, val in zip(self.vars[:dim], point)})
            H_numeric = np.array(H_eval).astype(float)
            
            # Compute eigenvalues to determine type
            eigvals = np.linalg.eigvals(H_numeric)
            
            if all(ev > 0 for ev in eigvals):
                return 'minimum'
            elif all(ev < 0 for ev in eigvals):
                return 'maximum'
            elif any(ev > 0 for ev in eigvals) and any(ev < 0 for ev in eigvals):
                return 'saddle'
            else:
                return 'unknown'
        except:
            return 'unknown'

class VerificationSuite:
    """Provides methods for verifying critical points and other analytical results."""
    
    @staticmethod
    def numerical_gradient_check(f_expr, point, h=1e-5):
        """
        Verify a critical point using numerical differentiation.
        
        Args:
            f_expr: SymPy expression representing the function
            point: Tuple representing the point coordinates
            h: Step size for numerical differentiation
            
        Returns:
            Boolean indicating if the point is a critical point
        """
        vars = sp.symbols('x y z')
        dim = len(point)
        f = sp.lambdify(vars[:dim], f_expr, 'numpy')
        
        # Calculate numerical gradient
        num_grad = []
        for i in range(dim):
            delta = np.zeros(dim)
            delta[i] = h
            forward = f(*(np.array(point) + delta))
            backward = f(*(np.array(point) - delta))
            num_grad.append((forward - backward) / (2 * h))
        
        # Check if gradient magnitude is close to zero
        return np.linalg.norm(num_grad) < 1e-5

class Visual3DEngine:
    """Handles visualization of functions, gradient fields, and critical points."""
    
    @staticmethod
    def plot_surface(f_expr, points=None):
        """
        Plot a 3D surface or contour plot of a function.
        
        Args:
            f_expr: SymPy expression representing the function
            points: List of critical points to highlight
        """
        vars = sp.symbols('x y z')
        dim = len(f_expr.free_symbols)
        f = sp.lambdify(vars[:dim], f_expr, 'numpy')

        # 2D function case
        if dim == 2:
            X = Y = np.linspace(-3, 3, 50)
            X, Y = np.meshgrid(X, Y)
            Z = f(X, Y)
            
            # Create 3D surface plot
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title(f'Surface: {str(f_expr)}')
            
            # Add critical points to surface plot
            if points:
                for pt in points:
                    try:
                        z_val = f(*pt)
                        ax1.scatter(pt[0], pt[1], z_val, color='red', s=50)
                    except:
                        pass
            
            # Create contour plot
            ax2 = fig.add_subplot(1, 2, 2)
            contour = ax2.contour(X, Y, Z, levels=20)
            ax2.clabel(contour, inline=True, fontsize=8)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title(f'Contour: {str(f_expr)}')
            
            # Add critical points to contour plot
            if points:
                x_points = [pt[0] for pt in points]
                y_points = [pt[1] for pt in points]
                ax2.scatter(x_points, y_points, color='red', s=30)
            
            plt.tight_layout()
            plt.show()
            
        # 3D function case
        elif dim == 3:
            # Create Plotly figure for isosurface
            X, Y, Z = np.mgrid[-3:3:20j, -3:3:20j, -3:3:20j]
            values = np.zeros(X.shape)
            
            # Compute function values
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k in range(X.shape[2]):
                        values[i,j,k] = f(X[i,j,k], Y[i,j,k], Z[i,j,k])
            
            fig = go.Figure(data=go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=values.flatten(),
                isomin=values.min(),
                isomax=values.max(),
                surface_count=5,
                colorscale='Viridis'
            ))
            
            # Add critical points
            if points:
                fig.add_trace(go.Scatter3d(
                    x=[pt[0] for pt in points],
                    y=[pt[1] for pt in points],
                    z=[pt[2] for pt in points],
                    mode='markers',
                    marker=dict(size=5, color='red')
                ))
            
            fig.update_layout(
                title=f"Isosurfaces of {str(f_expr)}",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                )
            )
            
            fig.show()

    @staticmethod
    def plot_gradient_field(f_expr):
        """
        Plot the gradient field of a function.
        
        Args:
            f_expr: SymPy expression representing the function
        """
        vars = sp.symbols('x y z')
        analyzer = Gradient3DAnalyzer()
        grad = analyzer.compute_gradient(f_expr)
        dim = len(grad)

        # 2D function case
        if dim == 2:
            X, Y = np.meshgrid(np.linspace(-3, 3, 15), np.linspace(-3, 3, 15))
            U = sp.lambdify(vars[:2], grad[0])(X, Y)
            V = sp.lambdify(vars[:2], grad[1])(X, Y)
            
            # Normalize vectors for better visualization
            magnitude = np.sqrt(U**2 + V**2)
            magnitude[magnitude == 0] = 1.0  # Avoid division by zero
            U_norm = U / magnitude
            V_norm = V / magnitude
            
            plt.figure(figsize=(8, 6))
            plt.quiver(X, Y, U_norm, V_norm, magnitude, cmap='viridis', pivot='mid')
            plt.colorbar(label='Gradient magnitude')
            plt.title(f'Gradient Field of {str(f_expr)}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            plt.show()

        # 3D function case
        elif dim == 3:
            X, Y, Z = np.mgrid[-3:3:5j, -3:3:5j, -3:3:5j]
            U = sp.lambdify(vars, grad[0])(X, Y, Z)
            V = sp.lambdify(vars, grad[1])(X, Y, Z)
            W = sp.lambdify(vars, grad[2])(X, Y, Z)

            # Create 3D vector field with Plotly
            fig = go.Figure(data=go.Cone(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                u=U.flatten(),
                v=V.flatten(),
                w=W.flatten(),
                colorscale='Blues',
                sizemode="absolute",
                sizeref=0.5
            ))
            
            fig.update_layout(
                title=f'Gradient Field of {str(f_expr)}',
                scene=dict(
                    aspectratio=dict(x=1, y=1, z=1),
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                )
            )
            
            fig.show()

class AIClassifier:
    """
    Machine learning classifier for critical points and symbolic query handling.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
        self.encoder = LabelEncoder()
        self.scope_terms = {'x', 'y', 'z', 'exp', 'sin', 'cos', 'log', 'sqrt', 'pi'}
        self.feature_size = 10  # Fixed feature vector size
        self.trained = False

    def generate_dataset(self, n_samples=80):
        """Generate a training dataset of functions and their critical points."""
        dataset = []
        
        # Basic quadratic functions (guaranteed examples)
        for a, b, c in [(1,1,'minimum'), (-1,-1,'maximum'), (1,-1,'saddle')]:
            f = f"{a}*x**2 + {b}*y**2"
            dataset.append({'function': f, 'label': c, 'critical_points': [(0,0)]})
        
        # Basic cubic functions
        for coeff in [1, -1]:
            f = f"{coeff}*x**3 + {coeff}*y**3 - 3*x*y"
            dataset.append({'function': f, 'label': 'saddle', 'critical_points': [(0,0)]})
        
        # Exponential functions
        f = f"exp(-(x**2 + y**2))"
        dataset.append({'function': f, 'label': 'maximum', 'critical_points': [(0,0)]})
        f = f"-exp(-(x**2 + y**2))"
        dataset.append({'function': f, 'label': 'minimum', 'critical_points': [(0,0)]})
        
        # 3D functions
        for f_str in ['x**2 + y**2 + z**2', 'x*y + y*z + z*x']:
            dataset.append({'function': f_str, 'label': 'minimum' if f_str[0] != 'x*' else 'saddle', 'critical_points': [(0,0,0)]})
        
        # Trigonometric functions
        for f_str in ['sin(x)*sin(y)', 'cos(x)*cos(y)']:
            dataset.append({'function': f_str, 'label': 'saddle', 'critical_points': [(0,0)]})
        
        # Generate remaining examples with polynomial combinations
        terms = ['x', 'y', 'x**2', 'y**2', 'x*y', 'x**3', 'y**3', 'x**2*y', 'x*y**2']
        
        while len(dataset) < n_samples:
            # Generate random function
            num_terms = np.random.randint(2, 5)
            selected_terms = np.random.choice(terms, num_terms, replace=False)
            coeffs = np.random.choice([-2, -1, 1, 2], num_terms)
            f_str = " + ".join([f"{c}*{t}" for c, t in zip(coeffs, selected_terms)])
            
            # Determine the type of critical point at origin
            try:
                f_expr = sp.sympify(f_str)
                analyzer = Gradient3DAnalyzer()
                grad = analyzer.compute_gradient(f_expr)
                
                # Check if origin is a critical point
                if all(g.subs({sp.Symbol('x'): 0, sp.Symbol('y'): 0}).evalf() == 0 for g in grad):
                    cp_type = analyzer.classify_critical_point(f_expr, (0, 0))
                    if cp_type != 'unknown':
                        dataset.append({'function': f_str, 'label': cp_type, 'critical_points': [(0,0)]})
            except:
                continue
            
        return dataset[:n_samples]

    def train(self, dataset=None):
        """Train the classifier using the provided or generated dataset."""
        if dataset is None:
            dataset = self.generate_dataset()
            
        print(f"Training on {len(dataset)} examples")
        
        # Extract features and labels
        X = [self._extract_features(d) for d in dataset]
        y = [d['label'] for d in dataset]
        self.encoder.fit(y)
        y_encoded = self.encoder.transform(y)
        
        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        accuracy = self.model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        self.trained = True
        
        return accuracy

    def _extract_features(self, data_point):
        """
        Extract numerical features from a function and its critical points.
        
        Args:
            data_point: Dictionary containing function and critical points
            
        Returns:
            List of numerical features for classification
        """
        f_expr = sp.sympify(data_point['function'])
        analyzer = Gradient3DAnalyzer()
        features = []
        
        # For each critical point, extract features
        for pt in data_point['critical_points']:
            dim = len(pt)
            H = analyzer.hessian(f_expr)
            
            try:
                # Evaluate Hessian at the critical point
                H_eval = H.subs({var: val for var, val in zip(analyzer.vars[:dim], pt)})
                H_numeric = np.array(H_eval).astype(float)
                eigvals = np.linalg.eigvals(H_numeric)
                
                # Extract features from critical point and Hessian
                pt_features = [
                    *pt[:2],  # x, y coordinates (pad if needed)
                    np.linalg.det(H_numeric) if H_numeric.size > 0 else 0,  # Determinant
                    np.trace(H_numeric) if H_numeric.size > 0 else 0,        # Trace
                    np.prod(np.sign(eigvals)) if eigvals.size > 0 else 0,    # Sign product of eigenvalues
                    len([v for v in eigvals if v > 0]) if eigvals.size > 0 else 0,  # Number of positive eigenvalues
                    len([v for v in eigvals if v < 0]) if eigvals.size > 0 else 0,  # Number of negative eigenvalues
                ]
                
                # Add features from the function itself
                function_str = str(f_expr)
                pt_features.extend([
                    1 if "**2" in function_str else 0,  # Has quadratic terms
                    1 if "**3" in function_str else 0,  # Has cubic terms
                    1 if any(t in function_str for t in ["sin", "cos", "exp"]) else 0  # Has transcendental functions
                ])
                
                # Ensure consistent feature vector size
                if len(pt_features) < self.feature_size:
                    pt_features += [0] * (self.feature_size - len(pt_features))
                else:
                    pt_features = pt_features[:self.feature_size]
                    
                features = pt_features
                
            except Exception as e:
                # Use default features in case of error
                features = [0] * self.feature_size
        
        return features

    def predict_with_uncertainty(self, function_str, point):
        """
        Predict the type of critical point with confidence measure.
        
        Args:
            function_str: String representation of the function
            point: Tuple representing the critical point coordinates
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.trained:
            self.train()
            
        try:
            features = self._extract_features({'function': function_str, 'critical_points': [point]})
            
            # Get prediction probabilities
            probs = self.model.predict_proba([features])[0]
            pred_class = self.encoder.inverse_transform([np.argmax(probs)])[0]
            confidence = np.max(probs)
            
            # Low confidence or high entropy indicates uncertainty
            if confidence < 0.7 or entropy(probs) > 0.5:
                return 'uncertain', confidence
                
            return pred_class, confidence
            
        except Exception as e:
            return 'uncertain', 0.0

    def is_within_scope(self, function_str):
        """Check if a function is within the classifier's scope."""
        try:
            f_expr = sp.sympify(function_str)
            atoms = f_expr.atoms(sp.Symbol, sp.Function)
            allowed = self.scope_terms
            return all(str(a) in allowed or a in sp.symbols('x y z') for a in atoms)
        except:
            return False

class MathAssistant:
    """
    Mathematical assistant that responds to user queries about gradient fields.
    """
    def __init__(self):
        self.analyzer = Gradient3DAnalyzer()
        self.classifier = AIClassifier()
        print("Initializing AI classifier...")
        self.classifier.train()
        self.examples = self._generate_examples()

    def _generate_examples(self):
        """Generate example functions for demonstration."""
        return [
            "x**2 + y**2",  # Simple minimum
            "-(x**2 + y**2)",  # Simple maximum
            "x**2 - y**2",  # Saddle point
            "exp(-(x**2 + y**2))",  # Gaussian peak
            "sin(x) * sin(y)",  # Periodic saddle points
            "x**3 + y**3 - 3*x*y"  # More complex function
        ]

    def respond(self, query):
        """
        Process and respond to user queries.
        
        Args:
            query: String representing the user's question or request
            
        Returns:
            String response to the query
        """
        query = query.strip().lower()
        
        # Help and examples
        if query in ('help', 'examples'):
            return self.example_usage()
        
        # Visualization requests
        if 'visualize' in query or 'plot' in query:
            return self.handle_visualization_request(query)
        
        # Critical point analysis
        if re.search(r'(is|does).*(point|min|max|saddle|critical)', query):
            return self.handle_analysis_query(query)
            
        # Find critical points
        if re.search(r'(find|locate|determine).*(critical|point|extrema)', query):
            return self.handle_critical_points_request(query)
        
        # Default response
        return (
            "I'm a gradient analysis assistant. You can ask me to:\n"
            "1. Analyze critical points (e.g., 'Is (0,0) a minimum for x²+y²?')\n"
            "2. Visualize functions (e.g., 'Visualize exp(-x²-y²)')\n"
            "3. Find critical points (e.g., 'Find critical points of x³+y³-3xy')\n"
            "Type 'examples' for more examples."
        )

    def example_usage(self):
        """Provide example queries for the user."""
        examples = [
            "Is (0,0) a minimum for x**2 + y**2?",
            "Does (0,0,0) represent a saddle point for x*y*z?",
            "Visualize exp(-(x**2 + y**2))",
            "Find critical points of x**3 + y**3 - 3*x*y",
            "Analyze sin(x)*sin(y) at (0,0)",
            "Help"
        ]
        
        return "Example queries:\n" + "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples))

    def handle_analysis_query(self, query):
        """Handle queries about classifying critical points."""
        # Try to extract function and point from query
        func_match = re.search(r'for\s+([^\(\?]+)', query)
        point_match = re.search(r'\(([^)]+)\)', query)
        
        if not func_match or not point_match:
            return "I couldn't understand your query. Please specify a function and a point, like: 'Is (0,0) a minimum for x**2 + y**2?'"
        
        func_str = func_match.group(1).strip()
        point_str = point_match.group(1).strip()
        
        try:
            # Convert point string to tuple of coordinates
            point = tuple(float(x.strip()) for x in point_str.split(',') if x.strip())
            
            # Check if function is within scope
            if not self.classifier.is_within_scope(func_str):
                return "Question is beyond my scope. I can analyze polynomial and basic transcendental functions."
            
            # Parse function
            f_expr = sp.sympify(func_str)
            
            # Verify if it's a critical point
            if not VerificationSuite.numerical_gradient_check(f_expr, point):
                return f"The point {point} is not actually a critical point of {func_str}."
            
            # Get classification and confidence
            pred, confidence = self.classifier.predict_with_uncertainty(func_str, point)
            
            # Get classical verification with Hessian
            analyzer = Gradient3DAnalyzer()
            H = analyzer.hessian(f_expr)
            
            try:
                dim = len(point)
                H_eval = H.subs({var: val for var, val in zip(analyzer.vars[:dim], point)})
                det = float(H_eval.det().evalf())
                
                # For 2D case, use second derivative test
                if dim == 2:
                    fxx = float(H_eval[0, 0].evalf())
                    classical_result = ""
                    if det > 0 and fxx > 0:
                        classical_result = "classical analysis confirms this is a minimum"
                    elif det > 0 and fxx < 0:
                        classical_result = "classical analysis confirms this is a maximum"
                    elif det < 0:
                        classical_result = "classical analysis confirms this is a saddle point"
                    else:
                        classical_result = "classical analysis is inconclusive (higher derivatives needed)"
                else:
                    classical_result = "classical analysis requires eigenvalue examination in higher dimensions"
            except:
                det = "could not compute"
                classical_result = "classical analysis failed"
            
            # Build response
            result = f"Analysis of {func_str} at point {point}:\n\n"
            
            if pred == 'uncertain':
                result += f"Classification: Uncertain (confidence too low: {confidence:.1%})\n"
            else:
                result += f"Classification: {pred} (confidence: {confidence:.1%})\n"
                
            result += f"Hessian determinant: {det}\n"
            result += f"Note: {classical_result}"
            
            return result
            
        except Exception as e:
            return f"Question is beyond my scope. Error: {str(e)}"

    def handle_visualization_request(self, query):
        """Handle function visualization requests."""
        # Extract function string
        func_match = re.search(r'(visualize|plot)\s+(.+?)(\s+with|\s*$)', query, re.I)
        
        if not func_match:
            return "Please specify a function to visualize, like: 'Visualize x**2 + y**2'"
        
        func_str = func_match.group(2).strip()
        
        try:
            # Check if function is within scope
            if not self.classifier.is_within_scope(func_str):
                return "Cannot visualize this function type. I handle polynomial and basic transcendental functions."
                
            # Parse function
            f_expr = sp.sympify(func_str)
            
            # Find critical points
            points = self.analyzer.find_critical_points(f_expr)
            
            # Generate visualizations
            Visual3DEngine.plot_surface(f_expr, points)
            Visual3DEngine.plot_gradient_field(f_expr)
            
            # Build response
            response = f"Visualized {func_str}"
            
            if points:
                response += f" with {len(points)} critical points: {points}"
                
                # Add classification for each point
                point_classifications = []
                for pt in points:
                    classification = self.analyzer.classify_critical_point(f_expr, pt)
                    point_classifications.append(f"{pt}: {classification}")
                    
                response += "\n\nCritical points classification:\n" + "\n".join(point_classifications)
            else:
                response += " (no critical points found in the search range)"
                
            return response
            
        except Exception as e:
            return f"Visualization failed. Please check function format. Error: {str(e)}"

    def handle_critical_points_request(self, query):
        """Handle requests to find critical points of a function."""
        # Extract function string
        func_match = re.search(r'of\s+(.+?)(\s+in|\s*$)', query)
        
        if not func_match:
            return "Please specify a function, like: 'Find critical points of x**2 - y**2'"
        
        func_str = func_match.group(1).strip()
        
        try:
            # Check if function is within scope
            if not self.classifier.is_within_scope(func_str):
                return "Question is beyond my scope. I can analyze polynomial and basic transcendental functions."
                
            # Parse function
            f_expr = sp.sympify(func_str)
            
            # Find critical points
            points = self.analyzer.find_critical_points(f_expr)
            
            # Build response
            if not points:
                return f"No critical points found for {func_str} in the search range (-5 to 5)."
            
            response = f"Critical points of {func_str}:\n\n"
            
            # Add classification for each point
            for pt in points:
                classification = self.analyzer.classify_critical_point(f_expr, pt)
                response += f"Point {pt}: {classification}\n"
                
            return response
            
        except Exception as e:
            return f"Question is beyond my scope. Error: {str(e)}"

def launch_assistant_cli():
    """Launch the mathematical assistant in a command-line interface."""
    assistant = MathAssistant()
    print("\nAI Gradient Analysis Assistant\n" + "="*30)
    print("Initialization complete. Type 'help' for usage examples.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ('exit', 'quit'):
                break
                
            response = assistant.respond(user_input)
            print("\nAI:", response)
        except KeyboardInterrupt:
            print("\nExiting gradient analysis assistant...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different query.")

def test_gradient_analysis():
    """Run a test suite for the gradient analysis functionality."""
    print("Running gradient analysis tests...")
    
    # Test functions
    test_functions = [
        (sp.sympify("x**2 + y**2"), [(0, 0)], 'minimum'),
        (sp.sympify("-(x**2 + y**2)"), [(0, 0)], 'maximum'),
        (sp.sympify("x**2 - y**2"), [(0, 0)], 'saddle'),
        (sp.sympify("x**3 + y**3 - 3*x*y"), [(0, 0), (1, 1)], 'saddle')
    ]
    
    analyzer = Gradient3DAnalyzer()
    
    # Test critical point finding
    for f_expr, expected_points, _ in test_functions:
        points = analyzer.find_critical_points(f_expr)
        print(f"Function: {f_expr}")
        print(f"  Expected points: {expected_points}")
        print(f"  Found points: {points}")
        
        # Check if expected points are found
        for expected_pt in expected_points:
            found = any(np.allclose(expected_pt, pt, rtol=1e-2) for pt in points)
            print(f"  Point {expected_pt} found: {found}")
    
    # Test classification
    for f_expr, points, expected_type in test_functions:
        for pt in points:
            classification = analyzer.classify_critical_point(f_expr, pt)
            print(f"Function: {f_expr} at point {pt}")
            print(f"  Expected type: {expected_type}")
            print(f"  Classified as: {classification}")
    
    print("Tests completed.")

def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gradient Field Analysis with AI Verification')
    parser.add_argument('--test', action='store_true', help='Run test suite')
    args = parser.parse_args()
    
    if args.test:
        test_gradient_analysis()
    else:
        launch_assistant_cli()

if __name__ == "__main__":
    main()