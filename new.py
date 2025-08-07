import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import time
import io
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(layout="wide", page_title="Feature Selection Comparison")

# Title and description
st.title("Comparative Analysis of Feature Selection Algorithms")
st.subheader("Correlation-Based vs Firefly Algorithm on Logistic Regression")

st.write("""
This application compares two feature selection methods:
- **Correlation-based feature selection**: Selects features based on their correlation with the target variable
- **Firefly Algorithm feature selection**: A nature-inspired optimization algorithm that selects optimal features
""")

# Data upload section
st.header("Dataset Selection")
data_option = st.radio(
    "Choose data source:",
    ("Upload your own dataset", "Use sample diabetes dataset")
)

if data_option == "Upload your own dataset":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Dataset successfully loaded!")
        except Exception as e:
            st.error(f"Error: {e}")
            # Fallback to sample data
            from sklearn.datasets import load_diabetes
            diabetes = load_diabetes(as_frame=True)
            data = diabetes.data
            data['target'] = diabetes.target
            # Convert target to binary for classification (threshold at median)
            median_target = data['target'].median()
            data['target'] = (data['target'] > median_target).astype(int)
            st.warning("Failed to load uploaded file. Using sample diabetes dataset instead.")
    else:
        # Default to sample data if no upload
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes(as_frame=True)
        data = diabetes.data
        data['target'] = diabetes.target
        # Convert target to binary for classification (threshold at median)
        median_target = data['target'].median()
        data['target'] = (data['target'] > median_target).astype(int)
        st.info("No file uploaded. Using sample diabetes dataset.")
else:
    # Load sample data
    try:
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes(as_frame=True)
        data = diabetes.data
        data['target'] = diabetes.target
        # Convert target to binary for classification (threshold at median)
        median_target = data['target'].median()
        data['target'] = (data['target'] > median_target).astype(int)
    except:
        # If sklearn dataset is not available, use synthetic data
        np.random.seed(42)
        n_samples = 442
        n_features = 10
        
        # Create synthetic features
        feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        data = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)
        
        # Create synthetic target with some correlation to features
        target = 0.3 * data['bmi'] + 0.2 * data['bp'] - 0.1 * data['s1'] + 0.4 * data['s2'] + np.random.randn(n_samples) * 0.5
        data['target'] = (target > target.median()).astype(int)
        st.info("Using synthetic diabetes dataset.")

# Target selection
st.subheader("Target Selection")
target_options = data.columns.tolist()
target_variable = st.selectbox("Select target variable:", target_options, index=target_options.index('target') if 'target' in target_options else len(target_options)-1)

# Convert target to binary if not already
if data[target_variable].nunique() > 2:
    st.warning(f"Target variable '{target_variable}' has {data[target_variable].nunique()} unique values. Converting to binary classification.")
    threshold_method = st.radio(
        "Choose binarization method:",
        ("Median threshold", "Custom threshold", "Use existing classes (0/1)"),
        index=0
    )
    
    if threshold_method == "Median threshold":
        threshold = data[target_variable].median()
        data[target_variable] = (data[target_variable] > threshold).astype(int)
        st.info(f"Converted to binary using median threshold: {threshold}")
    elif threshold_method == "Custom threshold":
        min_val = float(data[target_variable].min())
        max_val = float(data[target_variable].max())
        threshold = st.slider("Select threshold value:", min_val, max_val, (min_val + max_val)/2)
        data[target_variable] = (data[target_variable] > threshold).astype(int)
        st.info(f"Converted to binary using custom threshold: {threshold}")
    else:
        # Convert all non-zero values to 1
        data[target_variable] = (data[target_variable] != 0).astype(int)
        st.info("Using existing values as binary classes (0 and non-0)")

# Display dataset
with st.expander("Dataset Preview"):
    st.write(data.head())
    st.write(f"Dataset shape: {data.shape}")
    
    # Display target distribution
    col1, col2 = st.columns(2)
    with col1:
        st.write("Target Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(x=target_variable, data=data, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.write("Feature Correlation Matrix")
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, annot_kws={"size": 7})
        st.pyplot(fig)

# Define correlation-based feature selection
def correlation_based_selection(X, y, threshold=0.1):
    # Calculate correlation with target
    df = pd.DataFrame(X)
    df['target'] = y
    correlation = df.corr()['target'].drop('target')
    
    # Select features with correlation above threshold (absolute value)
    selected_features = correlation[abs(correlation) >= threshold].index.tolist()
    
    return selected_features, correlation

# Define Firefly Algorithm for feature selection
class FireflyAlgorithm:
    def __init__(self, n_fireflies=10, max_gen=20, alpha=0.5, beta0=1.0, gamma=1.0, n_features=None):
        self.n_fireflies = n_fireflies
        self.max_gen = max_gen
        self.alpha = alpha      # randomization parameter
        self.beta0 = beta0      # attractiveness at distance 0
        self.gamma = gamma      # light absorption coefficient
        self.n_features = n_features
        
    def initialize_population(self):
        # Initialize binary fireflies (1: feature selected, 0: feature not selected)
        return np.random.randint(0, 2, size=(self.n_fireflies, self.n_features))
    
    def distance(self, firefly_i, firefly_j):
        # Hamming distance between two binary strings
        return np.sum(firefly_i != firefly_j)
    
    def move_firefly(self, firefly_i, firefly_j):
        # Calculate attractiveness with distance
        r = self.distance(firefly_i, firefly_j)
        beta = self.beta0 * np.exp(-self.gamma * r**2)
        
        # Move firefly_i towards firefly_j
        for d in range(self.n_features):
            if firefly_i[d] != firefly_j[d]:
                if np.random.random() < beta:
                    firefly_i[d] = firefly_j[d]
            
            # Add randomization
            if np.random.random() < self.alpha:
                firefly_i[d] = 1 - firefly_i[d]  # flip bit
                
        return firefly_i
    
    def evaluate_fitness(self, X, y, solution):
        # If no features are selected, return very low fitness
        if np.sum(solution) == 0:
            return 0.0
        
        # Get selected features
        selected_features = np.where(solution == 1)[0]
        X_selected = X[:, selected_features]
        
        # Train logistic regression and evaluate
        try:
            model = LogisticRegression(max_iter=1000)
            scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
            fitness = np.mean(scores)
            
            # Add penalty for too many features (for parsimony)
            n_selected = len(selected_features)
            penalty = 0.01 * n_selected / self.n_features
            
            return fitness - penalty
        except:
            # If error occurs, return low fitness
            return 0.1
    
    def run(self, X, y):
        from sklearn.model_selection import cross_val_score
        
        # Initialize population
        fireflies = self.initialize_population()
        fitness = np.zeros(self.n_fireflies)
        
        # Evaluate initial fitness
        for i in range(self.n_fireflies):
            fitness[i] = self.evaluate_fitness(X, y, fireflies[i])
        
        # Main loop
        for g in range(self.max_gen):
            # Firefly movement
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if fitness[j] > fitness[i]:  # Move firefly i towards j if j is brighter
                        fireflies[i] = self.move_firefly(fireflies[i].copy(), fireflies[j])
                        # Re-evaluate fitness
                        fitness[i] = self.evaluate_fitness(X, y, fireflies[i])
        
        # Get best solution
        best_idx = np.argmax(fitness)
        best_solution = fireflies[best_idx]
        selected_features = np.where(best_solution == 1)[0]
        
        return selected_features.tolist(), best_solution, fitness[best_idx]

# Function to evaluate the model
def evaluate_model(X_train, X_test, y_train, y_test, selected_features, feature_names):
    # Get selected features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    # Train model
    start_time = time.time()
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_selected, y_train)
    train_time = time.time() - start_time
    
    # Predictions
    start_time = time.time()
    y_pred = model.predict(X_test_selected)
    pred_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate ROC curve and AUC
    y_prob = model.predict_proba(X_test_selected)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Get feature importance
    feature_importance = model.coef_[0]
    feature_importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in selected_features],
        'Importance': feature_importance
    })
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'train_time': train_time,
        'pred_time': pred_time,
        'feature_importance': feature_importance_df,
        'confusion_matrix': cm,
        'classification_report': report,
        'fpr': fpr,
        'tpr': tpr,
        'selected_features': [feature_names[i] for i in selected_features]
    }
    
    return results

# Main functionality
st.header("Feature Selection and Model Comparison")

# Parameters for feature selection
st.sidebar.header("Feature Selection Parameters")

# Correlation-based parameters
st.sidebar.subheader("Correlation-based Selection")
correlation_threshold = st.sidebar.slider(
    "Correlation Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.1, 
    step=0.05
)

# Firefly Algorithm parameters
st.sidebar.subheader("Firefly Algorithm")
n_fireflies = st.sidebar.slider("Number of Fireflies", 5, 30, 10)
max_gen = st.sidebar.slider("Maximum Generations", 5, 50, 15)
alpha = st.sidebar.slider("Alpha (Randomization)", 0.1, 1.0, 0.5, 0.1)
gamma = st.sidebar.slider("Gamma (Light Absorption)", 0.1, 2.0, 1.0, 0.1)

# Train/test split parameters
st.sidebar.subheader("Training Parameters")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
random_state = st.sidebar.slider("Random State", 0, 100, 42)

# Button to start analysis
if st.button("Run Analysis"):
    with st.spinner("Running feature selection and model evaluation..."):
        # Prepare data
        X = data.drop('target', axis=1).values
        y = data['target'].values
        feature_names = data.drop('target', axis=1).columns.tolist()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Correlation-based Feature Selection
        start_time = time.time()
        correlation_features, correlations = correlation_based_selection(
            pd.DataFrame(X_train, columns=feature_names), 
            y_train, 
            threshold=correlation_threshold
        )
        correlation_indices = [feature_names.index(feature) for feature in correlation_features]
        correlation_time = time.time() - start_time
        
        # Firefly Algorithm Feature Selection
        start_time = time.time()
        firefly = FireflyAlgorithm(
            n_fireflies=n_fireflies,
            max_gen=max_gen,
            alpha=alpha,
            gamma=gamma,
            n_features=X_train.shape[1]
        )
        firefly_features, firefly_solution, firefly_fitness = firefly.run(X_train, y_train)
        firefly_time = time.time() - start_time
        
        # Evaluate models
        correlation_results = evaluate_model(X_train, X_test, y_train, y_test, correlation_indices, feature_names)
        firefly_results = evaluate_model(X_train, X_test, y_train, y_test, firefly_features, feature_names)
        
        # Feature selection time comparison
        st.header("Feature Selection Time Comparison")
        fs_time_comparison = pd.DataFrame({
            'Method': ['Correlation-based', 'Firefly Algorithm'],
            'Time (seconds)': [correlation_time, firefly_time]
        })
        
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x='Method', y='Time (seconds)', data=fs_time_comparison, ax=ax)
        ax.set_title('Feature Selection Time Comparison')
        st.pyplot(fig)
        
        # Display selected features
        st.header("Selected Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation-based Features")
            st.write(f"Number of selected features: {len(correlation_features)}")
            st.write("Selected features:", correlation_features)
            
            # Show correlation values
            corr_df = pd.DataFrame({
                'Feature': feature_names,
                'Correlation': correlations
            }).sort_values('Correlation', key=abs, ascending=False)
            
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x='Correlation', y='Feature', data=corr_df, ax=ax)
            ax.set_title('Feature Correlation with Target')
            ax.axvline(x=0, color='black', linestyle='--')
            ax.axvline(x=correlation_threshold, color='red', linestyle='--')
            ax.axvline(x=-correlation_threshold, color='red', linestyle='--')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Firefly Algorithm Features")
            st.write(f"Number of selected features: {len(firefly_features)}")
            st.write("Selected features:", [feature_names[i] for i in firefly_features])
            
            # Show selected feature mask
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x=feature_names, y=firefly_solution, ax=ax)
            ax.set_title('Firefly Algorithm Feature Selection')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)
        
        # Performance metrics comparison
        st.header("Performance Metrics Comparison")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC',
                      'Training Time (s)', 'Prediction Time (s)'],
            'Correlation-based': [
                correlation_results['accuracy'],
                correlation_results['precision'],
                correlation_results['recall'],
                correlation_results['f1'],
                correlation_results['roc_auc'],
                correlation_results['train_time'],
                correlation_results['pred_time']
            ],
            'Firefly Algorithm': [
                firefly_results['accuracy'],
                firefly_results['precision'],
                firefly_results['recall'],
                firefly_results['f1'],
                firefly_results['roc_auc'],
                firefly_results['train_time'],
                firefly_results['pred_time']
            ]
        })
        
        st.table(metrics_df.set_index('Metric'))
        
        # Create comparison bar chart for main metrics
        main_metrics = metrics_df.iloc[:5].copy()
        main_metrics = pd.melt(main_metrics, id_vars=['Metric'], 
                               value_vars=['Correlation-based', 'Firefly Algorithm'],
                               var_name='Method', value_name='Value')
        
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x='Metric', y='Value', hue='Method', data=main_metrics, ax=ax)
        ax.set_title('Performance Metrics Comparison')
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        
        # Create comparison bar chart for time metrics
        time_metrics = metrics_df.iloc[5:].copy()
        time_metrics = pd.melt(time_metrics, id_vars=['Metric'], 
                              value_vars=['Correlation-based', 'Firefly Algorithm'],
                              var_name='Method', value_name='Value')
        
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x='Metric', y='Value', hue='Method', data=time_metrics, ax=ax)
        ax.set_title('Computational Time Comparison')
        st.pyplot(fig)
        
        # Confusion Matrices
        st.header("Confusion Matrices")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation-based Confusion Matrix")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(correlation_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix - Correlation-based')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Firefly Algorithm Confusion Matrix")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(firefly_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix - Firefly Algorithm')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
        
        # ROC Curves
        st.header("ROC Curves")
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # Correlation-based ROC
        ax.plot(correlation_results['fpr'], correlation_results['tpr'], 
                lw=2, label=f'Correlation-based (AUC = {correlation_results["roc_auc"]:.3f})')
        
        # Firefly Algorithm ROC
        ax.plot(firefly_results['fpr'], firefly_results['tpr'], 
                lw=2, label=f'Firefly Algorithm (AUC = {firefly_results["roc_auc"]:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
        # Feature Importance
        st.header("Feature Importance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation-based Feature Importance")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x='Importance', y='Feature', 
                       data=correlation_results['feature_importance'], ax=ax)
            ax.set_title('Feature Importance - Correlation-based')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Firefly Algorithm Feature Importance")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x='Importance', y='Feature', 
                       data=firefly_results['feature_importance'], ax=ax)
            ax.set_title('Feature Importance - Firefly Algorithm')
            st.pyplot(fig)
        
        # Summary and Conclusion
        st.header("Summary and Conclusion")
        
        accuracy_diff = abs(correlation_results['accuracy'] - firefly_results['accuracy'])
        
        if correlation_results['accuracy'] > firefly_results['accuracy']:
            better_method = "Correlation-based"
            worse_method = "Firefly Algorithm"
        else:
            better_method = "Firefly Algorithm"
            worse_method = "Correlation-based"
        
        st.write(f"""
        ### Performance Comparison
        
        - **{better_method}** selection achieved higher accuracy ({max(correlation_results['accuracy'], firefly_results['accuracy']):.4f}) 
          compared to **{worse_method}** ({min(correlation_results['accuracy'], firefly_results['accuracy']):.4f}), 
          a difference of {accuracy_diff:.4f}.
          
        - **Correlation-based** selection chose {len(correlation_features)} features, while 
          **Firefly Algorithm** selected {len(firefly_features)} features.
          
        - In terms of computational cost, **{min(correlation_time, firefly_time) < max(correlation_time, firefly_time) and 'Correlation-based' or 'Firefly Algorithm'}** 
          was faster for feature selection.
          
        ### Key Insights
        
        - Common features selected by both methods: 
          {set([feature_names[i] for i in firefly_features]).intersection(set(correlation_features))}
          
        - **{correlation_results['roc_auc'] > firefly_results['roc_auc'] and 'Correlation-based' or 'Firefly Algorithm'}** 
          had a better ROC AUC score ({max(correlation_results['roc_auc'], firefly_results['roc_auc']):.4f}).
          
        - For this specific diabetes classification task, the 
          **{better_method}** approach appears to be more suitable overall.
        """)
else:
    st.info("Set the parameters in the sidebar and click 'Run Analysis' to start the comparison.")

# Add explanations
with st.expander("Explanation of Methods"):
    st.write("""
    ### Correlation-based Feature Selection
    
    This method evaluates the correlation between each feature and the target variable. 
    Features with correlation magnitudes above a specified threshold are selected.
    
    **Advantages**:
    - Simple and computationally efficient
    - Easy to interpret
    - Works well when features are linearly related to the target
    
    **Disadvantages**:
    - Doesn't capture non-linear relationships
    - Doesn't account for feature interactions
    - Cannot identify redundant features
    
    ### Firefly Algorithm Feature Selection
    
    The Firefly Algorithm is a nature-inspired optimization method based on the flashing behavior of fireflies.
    
    **How it works**:
    1. Initialize population of fireflies (each representing a feature subset)
    2. Evaluate fitness (model performance) for each firefly
    3. Move less bright fireflies towards brighter ones
    4. Add randomization for exploration
    5. Iterate until convergence or maximum iterations
    
    **Advantages**:
    - Can find global optima
    - Can capture feature interactions
    - Works for non-linear relationships
    
    **Disadvantages**:
    - Computationally more expensive
    - Results may vary due to randomness
    - Requires parameter tuning
    """)

with st.expander("About Logistic Regression"):
    st.write("""
    ### Logistic Regression Classification
    
    Logistic Regression is a linear classification method that models the probability of a binary outcome.
    
    **Key characteristics**:
    - Uses a logistic function to transform linear predictions to probabilities
    - Good for binary classification problems
    - Provides interpretable coefficients (feature importance)
    - Efficient and fast to train
    
    **Performance metrics**:
    - **Accuracy**: Overall correctness of predictions
    - **Precision**: Proportion of positive identifications that were actually correct
    - **Recall**: Proportion of actual positives that were correctly identified
    - **F1-Score**: Harmonic mean of precision and recall
    - **ROC AUC**: Area under the ROC curve, measures discriminatory ability
    """)

st.sidebar.info("""
### About This App

This application demonstrates the comparative analysis of two feature selection methods:
1. Correlation-based selection
2. Firefly Algorithm optimization

Both methods are applied to a diabetes dataset, and their performance is evaluated using a Logistic Regression classifier.

Adjust the parameters in the sidebar and click 'Run Analysis' to see the results.
""")