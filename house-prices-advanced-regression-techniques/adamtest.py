from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
 # Load data with correct file paths
house_prices_train_data = pd.read_csv('C:/Users/enorton/Documents/Git/Kaggle/house-prices-advanced-regression-techniques/train.csv')
house_prices_test_data = pd.read_csv('C:/Users/enorton/Documents/Git/Kaggle/house-prices-advanced-regression-techniques/test.csv')

# First, let's identify categorical columns from our training data
categorical_cols = house_prices_train_data.select_dtypes(include=['object']).columns.tolist()
numeric_cols = house_prices_train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove 'Id' and 'SalePrice' from numeric columns if they exist
numeric_cols = [col for col in numeric_cols if col not in ['Id', 'SalePrice']]
# 3. Prepare training data
X_df = house_prices_train_data.drop(['SalePrice', 'Id'], axis=1)
y = house_prices_train_data['SalePrice']

# Handle missing values in test data
for col in test_df.select_dtypes(include=['float64', 'int64']).columns:
    if col in numeric_cols:  # Only fill if column exists in our numeric columns list
        test_df[col] = test_df[col].fillna(test_df[col].mean())

for col in test_df.select_dtypes(include=['object']).columns:
    if col in categorical_cols:  # Only fill if column exists in our categorical columns list
        test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

# Convert categorical variables to dummy variables
test_df = pd.get_dummies(test_df, drop_first=True)

# Ensure test data has same columns as training data
missing_cols = set(X_df.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0
test_df = test_df[X_df.columns]

def engineer_features(df):
    """Engineer advanced features for house price prediction"""
    df = df.copy()
    
    try:
        # Add new significant interactions
        df['GrLivArea_OverallQual'] = df['GrLivArea'] * df['OverallQual']
        df['OverallQual_GarageArea'] = df['OverallQual'] * df['GarageArea'].fillna(0)
        df['OverallQual_TotalBsmtSF'] = df['OverallQual'] * df['TotalBsmtSF'].fillna(0)
        df['GrLivArea_YearBuilt'] = df['GrLivArea'] * df['YearBuilt']
        df['OverallQual_YearBuilt'] = df['OverallQual'] * df['YearBuilt']
        
        # Total Square Footage
        df['TotalSF'] = df['BsmtFinSF1'].fillna(0) + df['BsmtFinSF2'].fillna(0) + \
                        df['BsmtUnfSF'].fillna(0) + df['TotalBsmtSF'].fillna(0) + \
                        df['1stFlrSF'] + df['2ndFlrSF'] + df['LowQualFinSF'] + df['GrLivArea']
        
        # House Age and Remodeling Features
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
        df['HouseAgeLessThan25'] = (df['HouseAge'] < 100).astype(int)
        
        # Age Effect
        df['AgeEffect'] = df['HouseAge'].apply(lambda x:
            -1551 * x + 235586 if x < 100 else 887 * x + 52089)
        df['AgeEffect_Normalized'] = (df['AgeEffect'] - df['AgeEffect'].mean()) / df['AgeEffect'].std()
        
        # Binary Features
        df['Has_Garage'] = (df['GarageType'].notna()).astype(int)
        df['Has_Basement'] = (df['BsmtQual'].notna()).astype(int)
        df['Is_Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
        df['Has_Pool'] = (df['PoolArea'] > 0).astype(int)
        df['Has_Fireplace'] = (df['Fireplaces'] > 0).astype(int)
        
        # Premium House Features
        df['IsHighQual'] = (df['OverallQual'] > 6).astype(int)
        df['HasSF'] = (df['TotalSF'] > 0).astype(int)
        df['PremiumHouse'] = df['TotalSF'] * ((df['IsHighQual'] == 1) & (df['HasSF'] == 1))
        
        # Polynomial Features
        df['TotalSF_Squared'] = df['TotalSF'] ** 2
        df['TotalSF_Cubed'] = df['TotalSF'] ** 3
        df['OverallQual_Squared'] = df['OverallQual'] ** 2
        df['OverallQual_Cubed'] = df['OverallQual'] ** 3
        df['YearBuilt_Squared'] = df['YearBuilt'] ** 2
        df['YearBuilt_Cubed'] = df['YearBuilt'] ** 3
        
        # Drop original columns that were transformed
        cols_to_drop = ['YearBuilt', 'YearRemodAdd']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
            
    except Exception as e:
        print(f"Warning during feature engineering: {str(e)}")
        
    return df

def handle_extreme_values(df):
    """Handle infinite and extreme values in the dataframe"""
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)
    
    for col in df_cleaned.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    return df_cleaned

def create_models(input_dim, optimal_lr):
    """Create different model architectures to compare"""
    models = {
        'Simple': Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1)
        ]),
        
        'Medium': Sequential([  # Our current architecture
            Dense(512, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1)
        ]),
        
        'Deep': Sequential([
            Dense(512, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1)
        ])
    }
    
    # Compile all models
    for model in models.values():
        model.compile(
            optimizer=Adam(learning_rate=optimal_lr),
            loss='mse'
        )
    
    return models

# Compare models using k-fold validation
def compare_architectures(X_scaled, y, n_folds=5):
    results = {}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Find optimal learning rate first
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2)
    optimal_lr = find_optimal_lr(X_train, y_train, X_val, y_val)
    
    # Create models with different architectures
    models = create_models(X_scaled.shape[1], optimal_lr)
    
    for name, model in models.items():
        print(f"\nEvaluating {name} architecture:")
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
            print(f"Fold {fold}/{n_folds}")
            
            X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            history = model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=100,
                batch_size=32,
                callbacks=[
                    EarlyStopping(patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6)
                ],
                verbose=0
            )
            
            # Record best validation score
            best_val_rmse = np.sqrt(min(history.history['val_loss']))
            fold_scores.append(best_val_rmse)
        
        # Calculate average performance
        results[name] = {
            'mean_rmse': np.mean(fold_scores),
            'std_rmse': np.std(fold_scores),
            'all_scores': fold_scores
        }
    
    # Print comparison results
    print("\nModel Comparison Results:")
    for name, metrics in results.items():
        print(f"\n{name} Architecture:")
        print(f"Mean RMSE: {metrics['mean_rmse']:.4f} ± {metrics['std_rmse']:.4f}")
    
    return results

# Use the comparison function
model_comparison = compare_architectures(X_scaled, y)

# Visualize results
plt.figure(figsize=(10, 6))
plt.boxplot([metrics['all_scores'] for metrics in model_comparison.values()], 
            labels=model_comparison.keys())
plt.title('Model Architecture Comparison')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def evaluate_metrics(y_true, y_pred, set_name="", fold=None, log_transformed=True):
    """Calculate multiple goodness of fit metrics"""
    if log_transformed:
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_true)

    metrics = {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'ExplainedVar': explained_variance_score(y_true, y_pred)
    }
    fold_info = f" - Fold {fold}" if fold is not None else ""
    print(f"\n{set_name}{fold_info} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    return metrics

# Load and prepare data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Prepare the data with log transformation
X = train_data.drop(['SalePrice', 'Id'], axis=1)
y = np.log1p(train_data['SalePrice'])  # Log transform target

# Engineer features
X = engineer_features(X)

# Handle missing values and encode categoricals
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].mean())
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Handle extreme values
X = handle_extreme_values(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def find_optimal_lr(X_train, y_train, X_val, y_val, start_lr=1e-6, end_lr=1e-1, num_points=20):
    """Find optimal learning rate using the learning rate range test"""
    print("Starting learning rate search...")
    
    # Generate fewer learning rates to test (20 instead of 50)
    learning_rates = np.geomspace(start_lr, end_lr, num_points)
    losses = []
    
    for i, lr in enumerate(learning_rates):
        print(f"Testing learning rate {i+1}/{num_points}: {lr:.2e}", end='\r')
        
        model = create_model(X_train.shape[1], learning_rate=lr)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=3,
            batch_size=32,
            verbose=0
        )
        losses.append(history.history['val_loss'][-1])
    
    # Find optimal learning rate
    smoothed_losses = np.array(losses)
    min_loss = np.min(smoothed_losses)
    optimal_lr_idx = np.gradient(smoothed_losses).argmin()
    optimal_lr = learning_rates[optimal_lr_idx]
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, losses)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    plt.axvline(x=optimal_lr, color='r', linestyle='--', label=f'Optimal LR: {optimal_lr:.2e}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"\nOptimal learning rate found: {optimal_lr:.2e}")
    return optimal_lr
    

def parameter_tuning(X_train, y_train, X_val, y_val, optimal_lr=1.44e-3):
    """Perform focused parameter tuning around the optimal learning rate"""
    print("Starting parameter tuning with focused search space...")
    
    # More focused parameter grid based on learning rate curve
    param_grid = {
        'learning_rate': [optimal_lr * 0.7, optimal_lr, optimal_lr * 1.3],  # Tight range around optimal
        'batch_size': [32, 64],  # Common effective batch sizes
        'dropout_rate': [0.3],    # Single value since this is less critical
        'l1_reg': [1e-5],        # Single value based on common practice
        'l2_reg': [1e-4]         # Single value based on common practice
    }
    
    best_val_loss = float('inf')
    best_params = None
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    current_combo = 0
    
    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            for dr in param_grid['dropout_rate']:
                for l1 in param_grid['l1_reg']:
                    for l2 in param_grid['l2_reg']:
                        current_combo += 1
                        print(f"\rTesting combination {current_combo}/{total_combinations}", end='')
                        
                        model = Sequential([
                            Dense(512, activation='relu', input_dim=X_train.shape[1],
                                  kernel_regularizer=l1_l2(l1=l1, l2=l2)),
                            BatchNormalization(),
                            Dropout(dr),
                            Dense(256, activation='relu',
                                  kernel_regularizer=l1_l2(l1=l1, l2=l2)),
                            BatchNormalization(),
                            Dropout(dr),
                            Dense(1)
                        ])
                        
                        optimizer = Adam(learning_rate=lr)
                        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
                        
                        # Quick evaluation with early stopping
                        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        
                        history = model.fit(
                            X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=30,  # Reduced epochs
                            batch_size=bs,
                            callbacks=[early_stop],
                            verbose=0
                        )
                        
                        val_loss = min(history.history['val_loss'])  # Get best validation loss
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = {
                                'lr': lr,
                                'batch_size': bs,
                                'dropout_rate': dr,
                                'l1_reg': l1,
                                'l2_reg': l2
                            }
    
    print("\n\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return best_params

# Modify the training loop to use optimal parameters
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Find optimal learning rate
print("Finding optimal learning rate...")
optimal_lr = find_optimal_lr(X_train, y_train, X_val, y_val)

# Perform parameter tuning
print("\nPerforming parameter tuning...")
best_params = parameter_tuning(X_train, y_train, X_val, y_val)
print("\nBest parameters found:")
print(best_params)

# Use the optimal parameters in k-fold cross validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
    print(f"\nFold {fold}")
    
    X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = create_model(
        input_dim=X_scaled.shape[1],
        learning_rate=best_params['lr']
    )
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_fold_train, y_fold_train,
        validation_data=(X_fold_val, y_fold_val),
        epochs=100,
        batch_size=best_params['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    
    # Evaluate fold
    train_pred = model.predict(X_fold_train).flatten()
    val_pred = model.predict(X_fold_val).flatten()
    
    train_metrics = evaluate_metrics(y_fold_train, train_pred, "Training", fold)
    val_metrics = evaluate_metrics(y_fold_val, val_pred, "Validation", fold)
    
    fold_results.append({
        'fold': fold,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'history': history.history
    })

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for fold_data in fold_results:
    plt.plot(fold_data['history']['loss'], label=f"Train Fold {fold_data['fold']}")
    plt.plot(fold_data['history']['val_loss'], label=f"Val Fold {fold_data['fold']}")
plt.title('Model Loss Across Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
train_rmse = [fold['train_metrics']['RMSE'] for fold in fold_results]
val_rmse = [fold['val_metrics']['RMSE'] for fold in fold_results]
plt.boxplot([train_rmse, val_rmse], labels=['Training', 'Validation'])
plt.title('RMSE Distribution Across Folds')
plt.ylabel('RMSE')

plt.tight_layout()
plt.show()

# Prepare test data for submission
test_df = engineer_features(test_data.drop(['Id'], axis=1))

# Handle missing values and encode categoricals for test data
for col in numeric_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(X[col].mean())
for col in categorical_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(X[col].mode()[0])

test_df = pd.get_dummies(test_df, columns=[col for col in categorical_cols if col in test_df.columns])

# Ensure test data has same columns as training data
missing_cols = set(X.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0
test_df = test_df[X.columns]

# Handle extreme values and scale test data
test_df = handle_extreme_values(test_df)
test_scaled = scaler.transform(test_df)

# Make predictions on test data
test_predictions = model.predict(test_scaled).flatten()
test_predictions = np.expm1(test_predictions)  # Transform back from log scale

# Create submission file
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

# Save submission
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_file = f'submission_deep_learning_{timestamp}.csv'
submission.to_csv(submission_file, index=False)

print(f"\nSubmission file created: {submission_file}")
print("\nModel Summary:")
model.summary()
