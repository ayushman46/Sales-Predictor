# Sales Prediction Model using Machine Learning
# This script predicts sales based on advertising spend across TV, Radio, and Newspaper

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SalesPredictionModel:
    """
    A comprehensive sales prediction model that handles data loading,
    exploration, preprocessing, model training, and evaluation.
    """
    
    def __init__(self, data_path=None):
        """Initialize the sales prediction model"""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        self.data_path = data_path or 'Advertising.csv'
        
    def load_data(self, file_path=None):
        """
        Load the advertising dataset
        If no file path provided, creates sample data for demonstration
        """
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                print("✅ Data loaded successfully from file!")
            except FileNotFoundError:
                print("❌ File not found. Creating sample data instead...")
                self.create_sample_data()
        else:
            print("📊 Creating sample advertising data for demonstration...")
            self.create_sample_data()
            
        return self.data
    
    def create_sample_data(self):
        """Create sample advertising data for demonstration purposes"""
        np.random.seed(42)
        n_samples = 200
        
        # Generate realistic advertising spend data
        tv_spend = np.random.normal(150, 50, n_samples)
        radio_spend = np.random.normal(25, 10, n_samples)
        newspaper_spend = np.random.normal(30, 15, n_samples)
        
        # Create realistic sales based on advertising spend with some noise
        # TV has highest impact, Radio moderate, Newspaper lowest
        sales = (0.045 * tv_spend + 
                0.188 * radio_spend + 
                0.001 * newspaper_spend + 
                np.random.normal(0, 2, n_samples) + 2)
        
        # Ensure no negative values
        tv_spend = np.maximum(tv_spend, 0)
        radio_spend = np.maximum(radio_spend, 0)
        newspaper_spend = np.maximum(newspaper_spend, 0)
        sales = np.maximum(sales, 0)
        
        self.data = pd.DataFrame({
            'TV': tv_spend,
            'Radio': radio_spend,
            'Newspaper': newspaper_spend,
            'Sales': sales
        })
        
        print(f"📈 Sample dataset created with {n_samples} records")
    
    def explore_data(self):
        """Perform comprehensive data exploration and visualization"""
        print("\n" + "="*60)
        print("📊 DATA EXPLORATION & ANALYSIS")
        print("="*60)
        
        # Basic information about the dataset
        print("\n🔍 Dataset Overview:")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Display first few rows
        print("\n📋 First 5 rows of the dataset:")
        print(self.data.head())
        
        # Statistical summary
        print("\n📈 Statistical Summary:")
        print(self.data.describe().round(2))
        
        # Check for missing values
        print("\n❓ Missing Values:")
        missing_values = self.data.isnull().sum()
        if missing_values.sum() == 0:
            print("✅ No missing values found!")
        else:
            print(missing_values)
        
        # Data types
        print("\n📝 Data Types:")
        print(self.data.dtypes)
        
        # Create visualizations
        self.create_visualizations()
        
        # Correlation analysis
        self.analyze_correlations()
    
    def create_visualizations(self):
        """Create comprehensive visualizations for data understanding"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sales Prediction - Data Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Distribution of Sales
        axes[0, 0].hist(self.data['Sales'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Sales', fontweight='bold')
        axes[0, 0].set_xlabel('Sales')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. TV vs Sales scatter plot
        axes[0, 1].scatter(self.data['TV'], self.data['Sales'], alpha=0.6, color='red')
        axes[0, 1].set_title('TV Advertising vs Sales', fontweight='bold')
        axes[0, 1].set_xlabel('TV Advertising Spend')
        axes[0, 1].set_ylabel('Sales')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Radio vs Sales scatter plot
        axes[0, 2].scatter(self.data['Radio'], self.data['Sales'], alpha=0.6, color='green')
        axes[0, 2].set_title('Radio Advertising vs Sales', fontweight='bold')
        axes[0, 2].set_xlabel('Radio Advertising Spend')
        axes[0, 2].set_ylabel('Sales')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Newspaper vs Sales scatter plot
        axes[1, 0].scatter(self.data['Newspaper'], self.data['Sales'], alpha=0.6, color='orange')
        axes[1, 0].set_title('Newspaper Advertising vs Sales', fontweight='bold')
        axes[1, 0].set_xlabel('Newspaper Advertising Spend')
        axes[1, 0].set_ylabel('Sales')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Box plot of all advertising channels
        advertising_data = self.data[['TV', 'Radio', 'Newspaper']]
        axes[1, 1].boxplot([advertising_data['TV'], advertising_data['Radio'], 
                           advertising_data['Newspaper']], labels=['TV', 'Radio', 'Newspaper'])
        axes[1, 1].set_title('Advertising Spend Distribution', fontweight='bold')
        axes[1, 1].set_ylabel('Spend Amount')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Correlation heatmap
        correlation_matrix = self.data.corr()
        im = axes[1, 2].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 2].set_title('Correlation Matrix', fontweight='bold')
        axes[1, 2].set_xticks(range(len(correlation_matrix.columns)))
        axes[1, 2].set_yticks(range(len(correlation_matrix.columns)))
        axes[1, 2].set_xticklabels(correlation_matrix.columns, rotation=45)
        axes[1, 2].set_yticklabels(correlation_matrix.columns)
        
        # Add correlation values to heatmap
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = axes[1, 2].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_correlations(self):
        """Analyze correlations between variables"""
        print("\n🔗 CORRELATION ANALYSIS:")
        print("-" * 40)
        
        correlation_matrix = self.data.corr()
        print("Correlation with Sales:")
        sales_correlations = correlation_matrix['Sales'].sort_values(ascending=False)
        
        for feature, corr in sales_correlations.items():
            if feature != 'Sales':
                strength = self.interpret_correlation(abs(corr))
                direction = "positive" if corr > 0 else "negative"
                print(f"  • {feature}: {corr:.3f} ({strength} {direction} correlation)")
    
    def interpret_correlation(self, corr_value):
        """Interpret correlation strength"""
        if corr_value >= 0.7:
            return "strong"
        elif corr_value >= 0.5:
            return "moderate"
        elif corr_value >= 0.3:
            return "weak"
        else:
            return "very weak"
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for machine learning"""
        print("\n" + "="*60)
        print("🔧 DATA PREPARATION")
        print("="*60)
        
        # Separate features and target variable
        X = self.data[['TV', 'Radio', 'Newspaper']]
        y = self.data['Sales']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {self.X_train.shape[0]} samples")
        print(f"Testing set size: {self.X_test.shape[0]} samples")
        
        # Scale the features for better model performance
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Data preparation completed!")
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n" + "="*60)
        print("🤖 MODEL TRAINING")
        print("="*60)
        
        # 1. Linear Regression
        print("\n🔵 Training Linear Regression Model...")
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = lr_model
        
        # 2. Random Forest Regressor
        print("🌲 Training Random Forest Model...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        
        print("✅ All models trained successfully!")
        
        # Display feature importance for Linear Regression
        self.display_feature_importance()
    
    def display_feature_importance(self):
        """Display feature importance from the models"""
        print("\n📊 FEATURE IMPORTANCE ANALYSIS:")
        print("-" * 40)
        
        # Linear Regression coefficients
        lr_model = self.models['Linear Regression']
        feature_names = ['TV', 'Radio', 'Newspaper']
        
        print("Linear Regression Coefficients:")
        for feature, coef in zip(feature_names, lr_model.coef_):
            print(f"  • {feature}: {coef:.4f}")
        print(f"  • Intercept: {lr_model.intercept_:.4f}")
        
        # Random Forest feature importance
        rf_model = self.models['Random Forest']
        print("\nRandom Forest Feature Importance:")
        for feature, importance in zip(feature_names, rf_model.feature_importances_):
            print(f"  • {feature}: {importance:.4f}")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("📈 MODEL EVALUATION")
        print("="*60)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🔍 Evaluating {model_name}:")
            print("-" * 30)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Store results
            results[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'predictions': y_pred
            }
            
            # Display metrics
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
            print(f"Model Accuracy: {r2*100:.2f}%")
        
        # Find the best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
        print(f"\n🏆 Best performing model: {best_model_name} (R² = {results[best_model_name]['R²']:.4f})")
        
        # Create evaluation plots
        self.create_evaluation_plots(results)
        
        return results
    
    def create_evaluation_plots(self, results):
        """Create plots to visualize model performance"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Performance Evaluation', fontsize=16, fontweight='bold')
        
        # Plot 1: Actual vs Predicted for each model
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (model_name, result) in enumerate(results.items()):
            axes[0].scatter(self.y_test, result['predictions'], 
                          alpha=0.6, label=model_name, color=colors[i])
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), min([r['predictions'].min() for r in results.values()]))
        max_val = max(self.y_test.max(), max([r['predictions'].max() for r in results.values()]))
        axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.75, zorder=0)
        
        axes[0].set_xlabel('Actual Sales')
        axes[0].set_ylabel('Predicted Sales')
        axes[0].set_title('Actual vs Predicted Sales')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Model comparison metrics
        metrics = ['R²', 'RMSE', 'MAE']
        model_names = list(results.keys())
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            axes[1].bar(x + i*width, values, width, label=metric, alpha=0.8)
        
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('Metric Values')
        axes[1].set_title('Model Performance Comparison')
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(model_names)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_sales(self, tv_spend, radio_spend, newspaper_spend, model_name='Linear Regression'):
        """Make predictions for new advertising spend values"""
        if model_name not in self.models:
            print(f"❌ Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            return None
        
        model = self.models[model_name]
        
        # Create input array
        input_data = np.array([[tv_spend, radio_spend, newspaper_spend]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        print(f"\n💰 SALES PREDICTION using {model_name}:")
        print("-" * 40)
        print(f"TV Spend: ${tv_spend:,.2f}")
        print(f"Radio Spend: ${radio_spend:,.2f}")
        print(f"Newspaper Spend: ${newspaper_spend:,.2f}")
        print(f"Predicted Sales: ${prediction:,.2f}")
        
        return prediction
    
    def generate_insights(self):
        """Generate business insights from the analysis"""
        print("\n" + "="*60)
        print("💡 BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Correlation insights
        correlation_matrix = self.data.corr()
        sales_correlations = correlation_matrix['Sales'].drop('Sales').sort_values(ascending=False)
        
        print("\n🎯 Key Findings:")
        print("-" * 20)
        
        strongest_channel = sales_correlations.index[0]
        strongest_corr = sales_correlations.iloc[0]
        
        print(f"1. {strongest_channel} advertising shows the strongest correlation with sales ({strongest_corr:.3f})")
        print(f"2. Average sales: ${self.data['Sales'].mean():.2f}")
        print(f"3. Sales range: ${self.data['Sales'].min():.2f} - ${self.data['Sales'].max():.2f}")
        
        # Model performance insights
        print("\n📊 Model Performance:")
        print("-" * 20)
        best_model = max(self.models.keys(), key=lambda x: self.models[x].score(self.X_test, self.y_test))
        print(f"• Best performing model: {best_model}")
        print(f"• Model can explain {self.models[best_model].score(self.X_test, self.y_test)*100:.1f}% of sales variance")
        
        print("\n🚀 Recommendations:")
        print("-" * 20)
        print(f"1. Focus more budget on {strongest_channel} advertising for maximum impact")
        print("2. Use the trained model to optimize advertising spend allocation")
        print("3. Monitor sales performance and retrain model with new data regularly")
        print("4. Consider testing different advertising strategies based on model predictions")
    
    def get_user_budget(self):
        """Get advertising budget input from user"""
        print("\n" + "="*60)
        print("💰 BUDGET INPUT")
        print("="*60)
        print("Please enter your advertising budget (in dollars)")
        print("Format: Enter numbers only, no symbols needed")
        
        try:
            tv = float(input("TV Advertising Budget: "))
            radio = float(input("Radio Advertising Budget: "))
            newspaper = float(input("Newspaper Advertising Budget: "))
            return {"TV": tv, "Radio": radio, "Newspaper": newspaper, "name": "Custom Budget"}
        except ValueError:
            print("❌ Invalid input! Please enter numbers only.")
            return None

def main():
    """Main function to run the complete sales prediction analysis"""
    print("🎯 SALES PREDICTION USING MACHINE LEARNING")
    print("=" * 80)
    print("This comprehensive analysis will help predict sales based on advertising spend")
    print("across TV, Radio, and Newspaper channels.")
    print("=" * 80)
    
    # Initialize the model
    predictor = SalesPredictionModel()
    
    # Load data from Advertising.csv
    predictor.load_data('Advertising.csv')
    
    # Explore the data
    predictor.explore_data()
    
    # Prepare data for machine learning
    predictor.prepare_data()
    
    # Train models
    predictor.train_models()
    
    # Evaluate models
    results = predictor.evaluate_models()
    
    # Generate business insights
    predictor.generate_insights()
    
    # Example predictions
    print("\n" + "="*60)
    print("🔮 EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Test different advertising scenarios
    scenarios = [
        {"TV": 100, "Radio": 20, "Newspaper": 10, "name": "Low Budget Campaign"},
        {"TV": 200, "Radio": 40, "Newspaper": 30, "name": "Medium Budget Campaign"},
        {"TV": 300, "Radio": 60, "Newspaper": 50, "name": "High Budget Campaign"}
    ]
    
    for scenario in scenarios:
        print(f"\n📺 {scenario['name']}:")
        predictor.predict_sales(scenario['TV'], scenario['Radio'], scenario['Newspaper'])
    
    # Get user's custom budget
    print("\n📊 Let's analyze your custom budget!")
    user_budget = predictor.get_user_budget()
    if user_budget:
        print(f"\n📺 Your Custom Budget Analysis:")
        predictor.predict_sales(user_budget['TV'], user_budget['Radio'], user_budget['Newspaper'])
    
    print("\n✅ Analysis completed successfully!")
    print("💡 You can now use this model to make informed decisions about advertising spend!")

if __name__ == "__main__":
    main()