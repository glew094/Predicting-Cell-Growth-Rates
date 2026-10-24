# Predictive Bioprocess Modeling of Specific Growth Rate

This repository contains a comprehensive machine learning pipeline designed to predict the specific growth rate of a cell culture based on historical fermentation data. By analyzing the complex, multidimensional interactions between temperature, pH, glucose concentration, and lactate concentration, this codebase solves the problem of inefficient, trial-and-error bioreactor tuning, allowing engineers to computationally optimize conditions for maximum cellular yield.

## Highlights

* Engineered a robust, modular machine learning pipeline for predictive bioprocess modeling.

* Prevented data leakage by embedding scaling transformations directly within Scikit-Learn pipelines.

* Optimized algorithm hyperparameters utilizing rigorous k-fold Grid Search Cross-Validation.

* Evaluated and contrasted parametric (Polynomial) and non-parametric (kNN) regression architectures.

* Visualized high-dimensional feature interactions using isolated partial dependence plotting techniques.

## Environment

* Python 3.11.x

* matplotlib 3.8.x

* numpy 1.26.x

* pandas 2.2.x

* scikit-learn 1.4.x

*(Note: Exact patch versions can be found in the accompanying `environment.yaml` file).*

## Code Design

The predictive workflow is executed sequentially via the `main.py` script. To ensure code maintainability, readability, and adherence to professional software engineering principles, the underlying mathematical and programmatic logic is modularized into dedicated helper functions stored within the `src/functions.py` file. The execution of the pipeline flows through five major structural stages, from initial data parsing to final metric extraction.

### 1. Initialization and Data Ingestion

**What does this stage do?**
The workflow begins by utilizing the standard `os` library to verify the existence of the `tables/` and `figures/` directories within the repository structure, automatically generating them if they are missing. Subsequently, the script utilizes `pandas` to read the raw historical fermentation dataset (`dataset_cell_growth_4factor.csv`). It maps the first column strictly to the target variable array (Specific Growth Rate) and the remaining columns to the input feature matrix (Temperature, pH, Glucose, and Lactate). Finally, it transforms these structured DataFrames into raw NumPy arrays.

**Why is this stage necessary?**
Checking for target directories is a fundamental defensive programming practice that prevents the entire script from crashing at the very end of its execution when it attempts to save output files. Furthermore, converting the raw CSV text into mathematically isolated arrays is computationally mandatory; supervised machine learning algorithms require absolute structural separation between the independent predictive inputs (features) and the dependent outcome (target) they are attempting to learn.

**How does this stage contribute to the overall workflow?**
This stage establishes the foundational digital environment. It ensures the local file system is primed to safely receive outputs and guarantees that the raw bioprocess data is cleaned, structured, and perfectly formatted for the rigorous mathematical transformations that occur in subsequent stages.

### 2. Data Partitioning

**What does this stage do?**
Using Scikit-Learn's `train_test_split` utility, this stage randomly divides the NumPy arrays into a training set (containing 80% of the data) and a testing set (containing 20% of the data). A fixed `random_state=0` parameter is explicitly applied to the function to ensure that this exact randomized split is reproducible across multiple independent executions of the program.

**Why is this stage necessary?**
A fundamental axiom of machine learning is that a model cannot be objectively evaluated on any data that it has already seen during its development. If a model is evaluated on its training data, the resulting error metrics will only reflect its ability to memorize those specific historical points (overfitting), not its true ability to understand the underlying biological relationships. Partitioning the data *before* any statistical scaling or processing occurs is an absolute necessity to prevent data leakage.

**How does this stage contribute to the overall workflow?**
By securely locking 20% of the historical data in an untouchable "testing vault," this stage guarantees that the final evaluation metrics produced at the end of the pipeline are highly objective, scientifically valid representations of how the model will actually perform when deployed on entirely new, unseen fermentation batches.

### 3. Pipeline Formulation and Preprocessing

**What does this stage do?**
This stage constructs two strictly encapsulated `make_pipeline` objects: one dictating the flow for Polynomial Regression, and another for k-Nearest Neighbors (kNN). Both pipelines strategically embed a `StandardScaler` as their very first operational step. This scaler applies a linear transformation to each individual feature, forcing them to possess a mean of 0 and a standard deviation of 1. The scaled data is then seamlessly passed forward to the `PolynomialFeatures` and `LinearRegression` modules, or the `KNeighborsRegressor` module, respectively.

**Why is this stage necessary?**
The raw bioprocess features operate on vastly different numerical scales; temperature might be measured in tens of degrees, while glucose and lactate concentrations might exist in single-digit millimolar units. Standardization is required to prevent variables with naturally larger numerical magnitudes from mathematically dominating the Euclidean distance calculations in kNN or improperly skewing the algebraic weights in polynomial regression. Furthermore, utilizing a pipeline securely binds the preprocessing steps directly to the estimators, severely reducing the risk of the training data's scaling parameters accidentally leaking into the test data folds.

**How does this stage contribute to the overall workflow?**
This stage standardizes the mathematical playing field. By encapsulating preprocessing within pipelines, it creates a robust, automated sequence that guarantees every single piece of data fed into the models—whether during cross-validation or final testing—is scaled identically and securely.

### 4. Hyperparameter Optimization (Grid Search CV)

**What does this stage do?**
This computational stage feeds the constructed pipelines into a `GridSearchCV` function alongside a dictionary of potential settings (Polynomial degrees 1 through 4; kNN neighbors 1 through 80, tested across both uniform and distance weights). The grid search employs a 5-fold cross-validation strategy, automatically subdividing the 80% training data into 5 smaller folds. It trains the algorithms on four folds and validates on the fifth, repeating this loop to calculate the average Root Mean Square Error (RMSE) for every single possible combination of hyperparameters.

**Why is this stage necessary?**
Machine learning algorithms possess parameters (which are learned automatically by the math) and hyperparameters (structural settings defined by the engineer). The optimal value of a hyperparameter cannot be easily learned during normal training. Engineers must manually iterate through these configurations to carefully balance the bias-variance tradeoff. For instance, a polynomial degree of 1 will severely underfit complex biological data (high bias), while a degree of 15 will drastically overfit the random noise (high variance). Cross-validation evaluates model performance across multiple permutations of the data, which heavily reduces the final metric's dependence on any one "lucky" data split.

**How does this stage contribute to the overall workflow?**
This stage acts as the primary tuning engine of the program. It rigorously searches the defined hyperparameter space to automatically locate the exact structural "sweet spot" that minimizes prediction error, returning two perfectly tuned, mathematically optimized models ready for the final testing phase.

### 5. Final Evaluation and Metric Extraction

**What does this stage do?**
The completely optimized Polynomial and kNN models are finally exposed to the unseen 20% test data previously isolated in Step 2. The script leverages these models to predict the target growth rates, compares these predictions to the true biological values, and calculates the final objective RMSE and Coefficient of Determination ($R^2$). These metrics are exported as a CSV summary table. Finally, the script extracts the standardized linear coefficients from the winning Polynomial model and generates an array of high-quality diagnostic figures, including validation curves, scaling comparisons, and feature importance charts.

**Why is this stage necessary?**
The evaluation metrics provide the empirical, statistical proof required to accurately determine which architectural approach (the parametric Polynomial model or the non-parametric kNN model) most successfully captured the underlying biology of the cells. Extracting the polynomial weights is necessary to achieve true model interpretability. Because the data was perfectly standardized before being passed through the polynomial regression step, we can directly read and interpret the model weights to understand exactly which environmental factors exert the strongest positive or negative influence on the specific growth rate.

**How does this stage contribute to the overall workflow?**
This final stage serves to translate abstract, highly mathematical models into concrete, actionable bioprocess logic. It provides biological operators and engineers with clear, quantified evidence of algorithmic performance and mechanistic drivers, fulfilling the core objective of the predictive pipeline.

## Analysis

### Justification of Model Architecture

The selection of the final predictive architecture relies on balancing mathematical complexity with generalizable accuracy, heavily informed by the bias-variance tradeoff. Through the implementation of `GridSearchCV`, the polynomial regression model was rigorously evaluated across degrees 1 through 4. The empirical evidence generated in the `poly_validation_curve.png` clearly demonstrates that a Polynomial Degree of 2 or 3 represents the optimal structural architecture. At Degree 1, the model severely underfits the data, unable to capture the natural, nonlinear saturation kinetics characteristic of biological cell growth (resulting in high training and high test RMSE). However, pushing the architecture to Degree 4 or higher forces the model into classic overfitting; the training error rapidly approaches zero as the equation memorizes random noise, while the validation error spikes exponentially. Therefore, a low-degree polynomial (Degree 2 or 3) is justified as it beautifully captures the smooth, continuous physical nature of biological temperature and pH dependencies without succumbing to variance.

### Comparative Analysis of Regression Algorithms

When comparing the chosen Polynomial model against the non-parametric k-Nearest Neighbors (kNN) algorithm, the Polynomial architecture demonstrates distinct advantages for this specific multidimensional bioprocess. As evidenced in `knn_validation_curve.png`, the kNN algorithm exhibits a steep error curve. At extremely low values of $k$ (e.g., $k=1$ or $2$), kNN drastically overfits the data, tracing individual, noisy data points rather than identifying global biological trends. While the kNN validation error does dip to a minimum as $k$ increases, it eventually climbs steadily as $k$ approaches 80 (underfitting), where the algorithm averages too many disparate conditions and simply predicts the global mean of the dataset.

Furthermore, kNN relies on calculating Euclidean distance across four dimensions (Temperature, pH, Glucose, Lactate). In high-dimensional spaces, distance-based metrics often degrade in predictive power—a phenomenon known as the "curse of dimensionality." Conversely, Polynomial Regression maps a continuous, mathematically defined surface over the data, which more accurately reflects the continuous nature of enzyme kinetics and cell metabolism, making it the mathematically superior alternative for this specific problem context.

### Advantages and Limitations of the Predictive Model

A primary advantage of the Polynomial Regression model is its high degree of interpretability. As seen in `model_weights.png`, because the features were processed via a `StandardScaler`, the resulting polynomial coefficients directly quantify the biological impact of each feature and their interacting terms. Engineers can instantly see exactly how strongly pH or Temperature drives the growth rate. Additionally, polynomial models are computationally inexpensive to deploy in real-time, executing basic algebraic equations rather than forcing a system to recalculate distances across thousands of historical training points (as required by kNN).

However, a major limitation of this polynomial architecture is its extreme vulnerability to extrapolation. Polynomial equations tend to diverge violently toward positive or negative infinity when fed input values that exist outside the boundaries of the original training data. If a bioreactor experiences a temperature spike far beyond the maximum temperature seen in the dataset, the model may predict an impossible, infinitely negative growth rate rather than a simple zero. It is strictly interpolative.

### Biological Influence of Process Parameters

The extracted model weights and generated partial dependence plots provide deep insight into how the four target parameters govern the specific growth rate:

* **Temperature:** This is typically the dominant driving force in cellular metabolism. The `temperature_dependence_curve.png` isolates this variable (holding others at their mathematical mean) to reveal a classic biological activity curve. As temperature rises, enzymatic activity and growth rate increase exponentially until an absolute thermal optimum is reached, after which the growth rate plummets rapidly due to sudden protein denaturation.

* **pH:** Cellular growth is highly sensitive to the concentration of hydrogen ions, which directly impact membrane potential and the structural integrity of transport proteins. The model identifies a tight, optimal pH band. Deviations to either highly acidic or highly basic conditions rapidly suppress the growth rate.

* **Glucose Concentration:** As the primary carbon and energy source, glucose exhibits a strong positive correlation with growth rate at lower concentrations, adhering to standard Monod kinetics. However, the model correctly captures that excessive glucose does not yield infinite growth; the cells eventually become saturated, and the growth rate plateaus regardless of additional glucose supplementation.

* **Lactate Concentration:** Lactate acts as a metabolic byproduct and a known cellular inhibitor. The polynomial weights accurately reflect a strong negative coefficient for lactate. As lactate accumulates in the media over the duration of a batch, it increases osmotic stress and lowers the local pH, cumulatively suppressing the specific growth rate.

### Optimal and Suboptimal Cell Culture Conditions

Based on the synthesized model, cell culture conditions that result in a highly optimal specific growth rate require maintaining a delicate equilibrium. The highest growth rates are achieved when the bioreactor temperature is held precisely at the peak of the thermal optimum curve (typically around 37°C for mammalian cells), accompanied by a strictly neutral or slightly optimized pH. Furthermore, glucose must be maintained at a sufficiently saturating level (to avoid starvation) while lactate concentrations must be kept as close to zero as physically possible to prevent metabolic inhibition.

Conversely, suboptimal conditions leading to low or zero specific growth rates occur when this equilibrium is broken. Significant drops in temperature will drastically slow down enzymatic kinetics, stalling growth. High temperatures will denature enzymes, rapidly killing the culture. Furthermore, late-stage fermentation conditions characterized by deeply depleted glucose reserves and heavy, toxic lactate accumulation will yield the lowest predicted growth rates in the dataset.

### Boundary Constraints and Model Reliability

The predictions generated by this polynomial model are highly reliable only when interpolating *within* the multidimensional boundaries of the historical training data. The model's predictions become exceedingly unreliable—and potentially mathematically impossible—if queried with extreme conditions. For example, if the model is asked to predict the growth rate at 100°C or at a pH of 1.0, the polynomial equation may output a massive positive or negative number due to the unbounded nature of high-degree algebraic curves. The model possesses no inherent biological knowledge; it does not "know" that cells die at boiling temperatures unless the training data explicitly maps out that boundary.

### Industrial Applications and Practical Implementation

In a practical industrial setting, this predictive machine learning model functions as the core logic for a "Digital Twin" or a soft sensor. By feeding real-time sensor data (Temperature, pH) and offline metabolite samples (Glucose, Lactate) into the deployed model, process engineers can instantly estimate the current specific growth rate without waiting for tedious, delayed offline cell counts.

Furthermore, this model can be integrated directly into automated Model Predictive Control (MPC) systems. If the model detects that the predicted growth rate is dropping due to rising lactate and falling glucose, it can automatically trigger a perfusion system to cycle in fresh media or initiate a precise, calculated glucose feed, ensuring the bioreactor operates at peak efficiency and maximum yield throughout the entire duration of the manufacturing campaign.