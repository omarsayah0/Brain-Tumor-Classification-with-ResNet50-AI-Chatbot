import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os
import json

def set_data():

    base_dir = "data"

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'train'),
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        directory=os.path.join(base_dir, 'val'),
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, val_generator

def set_model():

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,  
        input_shape=(256, 256, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(256, activation='relu'), 
        Dropout(0.2),
        Dense(4, activation='softmax') 
    ])

    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return (model)

def set_model2(model):

    base_model = model.layers[0] 
    base_model.trainable = True
    fine_tune_at = 70

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return (model)


def save_history(history, history_fine):

    history_data = {
        "accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "fine_tune_accuracy": history_fine.history["accuracy"],
        "fine_tune_val_accuracy": history_fine.history["val_accuracy"],
        "fine_tune_loss": history_fine.history["loss"],
        "fine_tune_val_loss": history_fine.history["val_loss"]
    }

    with open("training_history.json", "w") as f:
        json.dump(history_data, f)


def main():

    train_generator, val_generator = set_data()
    model = set_model()

    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )

    model = set_model2(model)

    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        initial_epoch=10
    )
    
    model.save("brain_tumor_resnet50_model.keras")
    save_history(history, history_fine)

if __name__ == "__main__":
    main()