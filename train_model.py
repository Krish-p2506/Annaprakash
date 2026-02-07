import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Dataset path
DATASET_PATH = r"D:\Annaprakash\data\dataset\lemon"

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32  # Increased from 8 for better stability (reduce to 16 if you get memory errors)
EPOCHS = 20      # Increased slightly because EarlyStopping will handle stopping

# 1. Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,  # Added shift for better robustness
    height_shift_range=0.2, # Added shift
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',    # Handles empty pixels after rotation
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
    shuffle=False # Important for correct evaluation metrics later
)

# 2. Build Model (Transfer Learning)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Freeze base model initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)  # Added Dropout to prevent overfitting
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# 3. Callbacks (The Safety Net)
checkpoint = ModelCheckpoint(
    "best_lemon_model.h5", 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)

callbacks_list = [checkpoint, early_stop, reduce_lr]

# 4. Phase 1: Train Head
print("--- Phase 1: Training Output Layers ---")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks_list
)

# 5. Phase 2: Fine-Tuning (Optional but Recommended)
print("--- Phase 2: Fine-Tuning Base Model ---")
base_model.trainable = True # Unfreeze the base model

# Freeze all layers except the last 30 (fine-tune only the top)
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a VERY low learning rate to avoid destroying learned weights
model.compile(
    optimizer=Adam(learning_rate=1e-5), 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10, # Add a few more epochs for fine-tuning
    callbacks=callbacks_list
)

# 6. Evaluation
print("--- Evaluation Report ---")
# Load best weights before evaluating
model.load_weights("best_lemon_model.h5") 

predictions = model.predict(val_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print(f"Final Model Saved as 'best_lemon_model.h5'")