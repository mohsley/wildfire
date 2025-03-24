import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling3D
from tensorflow.keras.layers import Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TrajectoryConvLSTM:
    def __init__(self, input_shape=(24, 224, 224, 3), output_path="model_output"):
        """
        Initialize the ConvLSTM model for trajectory analysis
        
        Parameters:
        - input_shape: (time_steps, height, width, channels)
        - output_path: Directory to save model outputs
        """
        self.input_shape = input_shape
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Build the model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build and compile the ConvLSTM model"""
        model = Sequential([
            # First ConvLSTM layer
            ConvLSTM2D(
                filters=64,
                kernel_size=(5, 5),
                padding='same',
                return_sequences=True,
                activation='relu',
                input_shape=self.input_shape
            ),
            BatchNormalization(),
            
            # Second ConvLSTM layer
            ConvLSTM2D(
                filters=64,
                kernel_size=(3, 3),
                padding='same',
                return_sequences=True,
                activation='relu'
            ),
            BatchNormalization(),
            MaxPooling3D(pool_size=(1, 2, 2)),
            
            # Third ConvLSTM layer
            ConvLSTM2D(
                filters=32,
                kernel_size=(3, 3),
                padding='same',
                return_sequences=False,  # Return only the last output
                activation='relu'
            ),
            BatchNormalization(),
            
            # Dense layers for classification/regression
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            # Output layer - adjust based on your task
            # For regression:
            Dense(1)
            # For classification:
            # Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model - adjust based on your task
        model.compile(
            optimizer='adam',
            loss='mse',  # For regression
            # loss='categorical_crossentropy',  # For classification
            metrics=['mae']  # For regression
            # metrics=['accuracy']  # For classification
        )
        
        model.summary()
        return model
    
    def prepare_dummy_data(self, num_samples=10):
        """
        Create dummy data for testing the model
        
        Parameters:
        - num_samples: Number of samples to generate
        
        Returns:
        - X: Input data (samples, time_steps, height, width, channels)
        - y: Target values
        """
        # Create random input data
        X = np.random.rand(num_samples, *self.input_shape)
        
        # Create random target values (regression)
        y = np.random.rand(num_samples, 1)
        
        # For classification:
        # num_classes = 3
        # y = np.random.randint(0, num_classes, size=(num_samples,))
        # y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
        
        return X, y
    
    def load_real_data(self, data_path):
        """
        Load real data from processed trajectory features
        
        Parameters:
        - data_path: Path to the numpy file with ConvLSTM input data
        
        Returns:
        - X: Input data
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        X = np.load(data_path)
        print(f"Loaded data with shape: {X.shape}")
        return X
    
    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=4):
        """
        Train the ConvLSTM model
        
        Parameters:
        - X: Input data
        - y: Target values
        - validation_split: Fraction of data to use for validation
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        
        Returns:
        - history: Training history
        """
        print(f"Training with {X.shape[0]} samples, {epochs} epochs, batch size {batch_size}")
        
        # Adjust the model output layer if needed based on y shape
        if hasattr(y, 'shape') and len(y.shape) > 1 and y.shape[1] > 1:
            output_dim = y.shape[1]
            if self.model.layers[-1].output_shape[-1] != output_dim:
                print(f"Adjusting output layer to match target dimension: {output_dim}")
                # Recreate the model with the correct output dimension
                self.model = Sequential([
                    layer for layer in self.model.layers[:-1]  # All layers except the last one
                ])
                self.model.add(Dense(output_dim))
                
                # Recompile the model
                self.model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )
        # Create a ModelCheckpoint callback
        checkpoint_path = os.path.join(self.output_path, "model_checkpoint.h5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True,
            mode='min'
        )
        
        # Create an EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping]
        )
        
        # Save the final model
        self.model.save(os.path.join(self.output_path, "final_model.h5"))
        
        # Plot training history
        self._plot_history(history)
        
        return history
    
    def _plot_history(self, history):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "training_history.png"))
        plt.close()
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Parameters:
        - X: Input data
        
        Returns:
        - predictions: Model predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance
        
        Parameters:
        - X: Input data
        - y: Target values
        
        Returns:
        - metrics: Evaluation metrics
        """
        return self.model.evaluate(X, y)

    def create_feature_maps(self, X, layer_names=None):
        """
        Visualize feature maps from ConvLSTM layers
        
        Parameters:
        - X: Input data (single sample)
        - layer_names: Names of layers to visualize (if None, use all ConvLSTM layers)
        """
        if layer_names is None:
            layer_names = [layer.name for layer in self.model.layers 
                          if isinstance(layer, ConvLSTM2D)]
        
        for layer_name in layer_names:
            # Create a model that outputs the feature maps
            feature_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(layer_name).output
            )
            
            # Get feature maps
            feature_maps = feature_model.predict(X)
            print(f"Feature maps shape for layer {layer_name}: {feature_maps.shape}")
            
            # Plot feature maps
            self._plot_feature_maps(feature_maps, layer_name)
    
    def _plot_feature_maps(self, feature_maps, layer_name):
        """Helper function to plot feature maps"""
        # For the first sample in the batch
        maps = feature_maps[0]
        
        # Get the time steps and number of filters
        time_steps, height, width, n_filters = maps.shape
        
        # Plot a selection of feature maps for the first and last time steps
        n_cols = 8  # Number of filters to show
        n_rows = 2  # First and last time step
        
        plt.figure(figsize=(20, 5))
        plt.suptitle(f'Feature Maps from Layer: {layer_name}', size=16)
        
        for i in range(min(n_cols, n_filters)):
            # First time step
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(maps[0, :, :, i], cmap='viridis')
            plt.title(f'Filter {i}, t=0')
            plt.axis('off')
            
            # Last time step
            plt.subplot(n_rows, n_cols, i + n_cols + 1)
            plt.imshow(maps[-1, :, :, i], cmap='viridis')
            plt.title(f'Filter {i}, t={time_steps-1}')
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.output_path, f"feature_maps_{layer_name}.png"))
        plt.close()

def create_training_data(X, future_steps=4):
    """
    Create training data for self-supervised learning.
    Uses part of the sequence to predict future steps.
    
    Parameters:
    - X: Input sequence data of shape [samples, time_steps, height, width, channels]
    - future_steps: Number of steps in the future to predict
    
    Returns:
    - X_train: Input sequences
    - y_train: Target values (future trajectory positions)
    """
    samples, time_steps, height, width, channels = X.shape
    
    # We'll use earlier steps to predict later steps
    input_steps = time_steps - future_steps
    
    X_train = []
    y_train = []
    
    # For each sample
    for s in range(samples):
        # Create input sequences (earlier time steps)
        X_train.append(X[s, :input_steps, :, :, :])
        
        # For targets, we'll use the trajectory path feature (channel 1)
        # from the last 'future_steps' time steps, flattened
        trajectory_feature = X[s, input_steps:, :, :, 1]  # Extract trajectory feature
        y_train.append(trajectory_feature.flatten())
    
    return np.array(X_train), np.array(y_train)

if __name__ == "__main__":
    # Load the processed data from the trajectory extraction script
    data_path = "convlstm_features/convlstm_input.npy"
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find processed data at {data_path}")
        print("Please run the trajectory-extraction.py script first to generate the features.")
        exit(1)
    
    # Load the data
    print(f"Loading data from {data_path}...")
    try:
        X = np.load(data_path)
        print(f"Loaded data with shape: {X.shape}")
        
        # Get the input shape from the loaded data
        _, time_steps, height, width, channels = X.shape
        input_shape = (time_steps, height, width, channels)
        
        # Initialize the model with the correct input shape
        print("Initializing ConvLSTM model...")
        trajectory_model = TrajectoryConvLSTM(input_shape)
        
        # Create training data using self-supervised approach
        print("Creating self-supervised training data...")
        future_steps = 4  # Predict the last 4 time steps
        X_train, y_train = create_training_data(X, future_steps)
        
        print(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
        
        # Train the model
        print("Training the model...")
        history = trajectory_model.train(X_train, y_train, epochs=30, batch_size=2)
        
        # Generate predictions for the first sample
        print("Generating predictions...")
        predictions = trajectory_model.predict(X_train[:1])
        print("Prediction shape:", predictions.shape)
        
        # Create feature map visualizations
        print("Creating feature map visualizations...")
        trajectory_model.create_feature_maps(X_train[:1])
        
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error loading or processing the data: {e}")
        import traceback
        traceback.print_exc()
        
        # If real data loading fails, fall back to dummy data
        print("Falling back to dummy data for demonstration...")
        input_shape = (24, 224, 224, 3)
        trajectory_model = TrajectoryConvLSTM(input_shape)
        X, y = trajectory_model.prepare_dummy_data(num_samples=16)
        history = trajectory_model.train(X, y, epochs=10, batch_size=4)